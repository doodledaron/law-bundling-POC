import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    MODEL_NAME = "google/gemini-2.5-flash-lite"  # served through OpenRouter
    GENERATION_CONFIG = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 40,
        # Raised from 1024: GLM spends part of this budget on hidden reasoning,
        # so dense pages were cut off before the transcription finished.
        # Raised again to 16384: a 13-page relevance call used all 8192 on
        # reasoning and returned no items. Reasoning cannot be disabled on GLM.
        "max_output_tokens": 16384
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