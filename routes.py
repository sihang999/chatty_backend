from flask import request, jsonify
from chatbot import model_call
from langchain_huggingface import HuggingFaceEmbeddings


# register router
def register_routes(app):
    #@app.route('/')if the website is homepage
    @app.route('/SihangRobot', methods=['POST'])
    def SihangRobot():
        try:
            # get the message from request
            data = request.get_json()
            # data.get(key, default) 方法的第二个参数是可选的，这里表示当键不存在时返回空。
            user_message = data.get("message", "")
            thread_id = data.get('sessionID', "")

            #检查输入是否正确？？？
            # print("---------------------------user input----------------------------------\n")
            # print(user_message)
            # print(session_id)

            if not user_message:
                return jsonify({"response": "Message cannot be empty"}), 400
            response_message = model_call(user_message, thread_id) 
            return jsonify({"response": response_message})

        except Exception as e:
            print(f"Error: {e}")
            return jsonify({"response": "An error occurred"}), 500
