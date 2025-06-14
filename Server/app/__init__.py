
from flask import Flask
from app.routes.donations import donations_bp
from app.routes.ai import ai_bp
from app.services.supabase_service import init_supabase

app = Flask(__name__)

# Initialize Supabase connection
init_supabase()

# Register Blueprints
app.register_blueprint(donations_bp, url_prefix='/donations')
app.register_blueprint(ai_bp, url_prefix='/ai')
