from flask import request, jsonify
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain
from app.utils import vector_store
from app.model import graph

# register router
def register_routes(app):
    #@app.route('/')if the website is homepage
    @app.route('/SihangRobot', methods=['POST'])
    def SihangRobot():
        global system_prompt_fmt
        try:
            # get the message from request
            user_message = request.json.get("message", "")
            # print("---------------------------user input----------------------------------\n")
            # print(user_message)
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


            # outmess = output["messages"]
            # print("---------------------------output1---------------------------------\n")
            # for m in outmess:
            #     m.pretty_print()
            # print("---------------------------output2----------------------------------\n")
            # print(output)

            response_message = output["messages"][-1].content
            return jsonify({"response": response_message})

        except Exception as e:
            print(f"Error: {e}")
            return jsonify({"response": "An error occurred"}), 500
