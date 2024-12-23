from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters.markdown import MarkdownTextSplitter
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, RemoveMessage
import json
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.runnables import chain
import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
#--------------------------------------------------------------------
folder_path = "./data/markdown_files"
status_file = "file_status.json"
vector_store_path = "faiss_index"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
#vector_store = InMemoryVectorStore(embeddings)
text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)


if os.path.exists(vector_store_path):
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )


# 文件状态管理（读取上次文件状态）
if os.path.exists(status_file):
    with open(status_file, "r") as f:
        file_status = json.load(f)
else:
    file_status = {}

# 记录新的文件状态
new_file_status = file_status.copy()
change=False

# 遍历文件夹中的 Markdown 文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".md"):
        file_path = os.path.join(folder_path, file_name)
        file_mtime = os.path.getmtime(file_path)  # 获取文件修改时间
        file_id = file_name  # 使用文件名作为唯一标识符

        # 检查文件是否是新的或已修改
        if file_id not in file_status or file_status[file_id]["mtime"] != file_mtime:
            print(f"Processing changed or new file: {file_name}")

            # 加载并分割文档
            loader = UnstructuredMarkdownLoader(file_path, mode="single", strategy="fast")
            docs = loader.load()
            split_docs = text_splitter.split_documents(docs)

            # 删除旧向量（如果文件已存在）
            if file_id in file_status:
                print(f"Deleting old vectors for {file_name}")
                old_uuids = file_status[file_id]["uuids"]
                vector_store.delete(ids=old_uuids)

            # 为新向量生成 UUID 并添加到向量存储
            new_uuids = [str(uuid4()) for _ in range(len(split_docs))]
            vector_store.add_documents(documents=split_docs, ids=new_uuids)
            print(f"Added {len(split_docs)} documents for {file_name}")

            # 更新文件状态
            new_file_status[file_id] = {
                "mtime": file_mtime,
                "uuids": new_uuids
            }
            change=True

        else:
            print(f"No changes detected for {file_name}")

        

if change:
    # 保存更新后的向量存储
    vector_store.save_local(vector_store_path)

    # 保存新的文件状态
    with open(status_file, "w") as f:
        json.dump(new_file_status, f)

    print("Update complete!")



#---------------------------------------------------------------------------------------------------





class State(MessagesState):
    summary: str

model = ChatOpenAI(model="gpt-4o", temperature=0.5)

system_prompt_fmt = ""

def call_model(state: State):
    
    # Get summary if it exists
    summary = state.get("summary", "")
    global system_prompt_fmt 

    # If there is summary, then we add it
    if summary:

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{system_prompt_fmt}. Summary: {summary}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        # Append summary to any newer messages
        message_prompt = prompt_template.invoke(state)
        
    else:

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{system_prompt_fmt}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        # Append summary to any newer messages
        message_prompt = prompt_template.invoke(state)

    # 调用模型
    print("---------------------------这是完整输入,原始----------------------------------\n")
    print(message_prompt)
    response = model.invoke(message_prompt)
    return {"messages": response}


# Define the logic to how to summarize conversation, which will be used for a "summarize_conversation" node
def summarize_conversation(state: State):
    summary = state.get("summary", "")

    # Create our summarization prompt，这个任务也是由chat Mode来做的。
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


# Determine whether to end or summarize the conversation/conditional edge
def should_continue(state: State):
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END

# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

#-----------------------------------------------------------------------------------------------------------------------------------


app = Flask(__name__)
CORS(app)

# 定义 Flask API 路由
@app.route('/api/chat', methods=['POST'])
def chat():
    global system_prompt_fmt
    try:
        # 从请求中获取消息
        user_message = request.json.get("message", "")
        print("---------------------------用户输入,容易阅读----------------------------------\n")
        print(user_message)
        if not user_message:
            return jsonify({"response": "Message cannot be empty"}), 400

        # Define a system prompt that tells the model how to use the retrieved context
        system_prompt = """You are an AI assistant representing Sihang Wu. Your primary role is to introduce Sihang Wu to others in a professional, accurate, and engaging manner based on the provided background information and context.
When responding:
- Focus on key details that highlight Sihang Wu's strengths, aspirations, and achievements.
- Tailor the introduction to the audience or context, ensuring relevance and clarity.
- Avoid sharing sensitive or irrelevant information unless explicitly prompted.
- If unsure about the context or details, politely ask clarifying questions or explain your limitations.
Your goal is to leave a positive, lasting impression of Sihang Wu by accurately reflecting their personality, skills, and accomplishments in a way that resonates with the audience.
The following context and earlier conversation summary will help you:
Context: {context}"""
# You are an AI assistant designed to assist Sihang Wu. You have access to all relevant documents provided by Sihang Wu and are required to use them to respond accurately and contextually to their needs. Your primary objectives are to:


# When responding:
# - Prioritize the accuracy and relevance of the information.
# - Use technical depth suitable for a master's student in Communication and Electronics Engineering.
# - Respect privacy and confidentiality of all information.

# You are permitted to ask clarifying questions if a request or document lacks sufficient detail. However, you must never invent information or act beyond the provided documents and your expertise.

        @chain
        def retriever(query: str) -> List[Document]:
            return vector_store.similarity_search(query, k=3)

        # Retrieve relevant documents
        docs = retriever.invoke(user_message)
        # Combine the documents into a single string
        docs_text = "".join(d.page_content for d in docs)
        # Populate the system prompt with the retrieved context
        system_prompt_fmt = system_prompt.format(context=docs_text)
        # Generate a response


        # Create a thread
        config = {"configurable": {"thread_id": "1"}}
        output = graph.invoke({"messages": [user_message]}, config) 
        #output = graph.invoke({"messages": [SystemMessage(content=system_prompt_fmt), HumanMessage(content=user_message)]}, config) 


        outmess = output["messages"]
        print("---------------------------这是完整输出,容易阅读----------------------------------\n")
        for m in outmess:
            m.pretty_print()
        print("---------------------------这是完整输出,原始----------------------------------\n")
        print(output)
        # 获取响应消息
        response_message = output["messages"][-1].content
        return jsonify({"response": response_message})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "An error occurred"}), 500

# 启动服务器
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


