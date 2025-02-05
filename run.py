from routes import register_routes
from update_index import manage_files, ensure_files_and_folders_exist
from datetime import datetime
from time import sleep
import threading
from flask import Flask
from flask_cors import CORS
import sentence_model

# # # get the enviroment variables
# from dotenv import load_dotenv
# load_dotenv()

# EMBEDDINGS_MODELv1 = None

def update_files_task():
    while True:
        now = datetime.now()
        if now.hour == 2:  
            ensure_files_and_folders_exist()
            manage_files()
        sleep(2700)  

# def get_model():
#     global EMBEDDINGS_MODELv1
#     EMBEDDINGS_MODELv1 = HuggingFaceEmbeddings(model_name="./multi-qa-mpnet-base-dot-v1")

if __name__ == '__main__':
    
    sentence_model.get_model()

    app = Flask(__name__)
    CORS(app)  
    register_routes(app)

    flask_thread = threading.Thread(target=update_files_task)
    flask_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=False)
    # app.run(host="0.0.0.0", port=5000, ssl_context=("cert.pem", "key.pem"))

