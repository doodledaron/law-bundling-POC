import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME = "gemini-2.0-flash"
    GENERATION_CONFIG = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024
    }
    
    # File upload settings
    MAX_FILE_SIZE = None  # No file size limit
    ALLOWED_EXTENSIONS = {
        'txt', 'pdf', 'doc', 'docx', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'
    }
    
    # Upload directory
    UPLOAD_DIR = "uploads"
    
    # API Authentication settings
    API_KEYS_STRING = os.getenv("API_KEYS", "")
    API_KEYS = [key.strip() for key in API_KEYS_STRING.split(",") if key.strip()] if API_KEYS_STRING else []
    
    # Authentication header name
    API_KEY_HEADER = "X-API-Key" 