from flask import Flask, request, jsonify
from flask_cors import CORS
#from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
#load_dotenv()

app = Flask(__name__)
CORS(app)

class State(MessagesState):
    summary: str

model = ChatOpenAI(model="gpt-4o-mini")


# Define the logic to call the model, which will be used for a "conversation" node
def call_model(state: State):
    
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:
        
        # Add system prompt and summary to system message不能以名字测试它的记忆！所有的state都是储存在运行内存的，重新运行就没有了！
        system_message = f"You are an assistant created by Sihang to answer questions about himself, there is summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    
    else:
        # Add system prompt
        system_message = f"You are an assistant created by Sihang to answer questions about himself."

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]

    # 调用模型
    response = model.invoke(messages)
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



# 定义 Flask API 路由
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # 从请求中获取消息
        user_message = request.json.get("message", "")
        if not user_message:
            return jsonify({"response": "Message cannot be empty"}), 400

        # Create a thread
        config = {"configurable": {"thread_id": "1"}}
        output = graph.invoke({"messages": [user_message]}, config) 

        # 获取响应消息
        response_message = output["messages"][-1].content
        return jsonify({"response": response_message})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "An error occurred"}), 500

# 启动服务器
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

