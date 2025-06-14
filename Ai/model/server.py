from flask import Flask
from app.routes.ai import ai_bp  # Import your blueprint

app = Flask(__name__)

# Register the AI route
app.register_blueprint(ai_bp, url_prefix='/api')

# Other blueprints can go here too
# app.register_blueprint(donations_bp, url_prefix='/api/donations')

if __name__ == '__main__':
    app.run(debug=True)
