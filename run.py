from app import create_app
from app.routes import register_routes

if __name__ == '__main__':
    app = create_app()
    register_routes(app)
    app.run(host='0.0.0.0', port=5000, debug=True)
