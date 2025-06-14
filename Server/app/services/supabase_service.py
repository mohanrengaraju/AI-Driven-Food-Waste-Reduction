# app/services/supabase.py
from supabase import create_client
import os
from dotenv import load_dotenv
from pathlib import Path

# 1. Get the absolute path to .env
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)


SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

supabase = None

def init_supabase():
    global supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase():
    global supabase
    return supabase
