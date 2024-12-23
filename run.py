from app import create_app
from app.routes import register_routes
from datetime import datetime
from app.utils import manage_files
from time import sleep
import threading


app = create_app()
register_routes(app)
def update_files_task():
    while True:
        now = datetime.now()
        if now.hour == 2:  
            manage_files()
        sleep(2700)  

if __name__ == '__main__':
        
    flask_thread = threading.Thread(target=update_files_task)
    flask_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=True)

