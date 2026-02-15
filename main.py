
"""
FastAPI application for law document processing system.
Handles HTTP endpoints and delegates processing to Celery tasks.

This is the entry point. All route handlers and business logic
are organized in the routes/ and services/ packages.
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

# Import routers
from routes.api_routes import api_router
from routes.dev_routes import dev_router
from routes.relevance_routes import relevance_router

# Import shared services (needed for health check)
from services.processing import (
    redis_client, get_parallel_processing_status, MAX_DOCUMENT_WORKERS
)
from tasks.utils import get_timestamp

# Initialize logging
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("chunks", exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Law Document Processing API",
    description="API for processing legal documents with OCR and information extraction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/results", StaticFiles(directory="results"), name="results")

# Include routers
app.include_router(api_router)
app.include_router(dev_router)
app.include_router(relevance_router)


# Root endpoint - API information page
@app.get("/", response_class=HTMLResponse)
async def root_info():
    """
    Display API information page without exposing development endpoints
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Law Document Processing API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .header { text-align: center; margin-bottom: 40px; }
            .api-info { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }
            .status { color: #28a745; font-weight: bold; }
            .endpoint { background: #e9ecef; padding: 10px; margin: 10px 0; border-radius: 4px; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏛️ Law Document Processing API</h1>
            <p class="status">Service Status: Active</p>
        </div>
        
        <div class="api-info">
            <h2>📋 API Information</h2>
            <p>This is a professional document processing service for legal documents.</p>
            
            <h3>Available Endpoints:</h3>
            <div class="endpoint">POST /api/upload - Document upload (requires API key)</div>
            <div class="endpoint">GET /api/job/{job_id} - Check processing status (requires API key)</div>
            <div class="endpoint">GET /health - Service health check (public)</div>
            
            <h3>Authentication:</h3>
            <p>All API endpoints require authentication via <code>X-API-Key</code> header.</p>
            
            <h3>Documentation:</h3>
            <p>For API integration documentation, please contact the system administrator.</p>
            
            <h3>Supported Formats:</h3>
            <p>PDF, JPEG, PNG files (no size limit)</p>
        </div>
        
        <div style="text-align: center; margin-top: 40px; color: #666;">
            <p>Law Document Processing Service v2.0.0</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    # Check Redis connection
    try:
        redis_client.ping()
        redis_status = "ok"
    except Exception as e:
        redis_status = f"error: {str(e)}"
    
    # Check parallel processing system status
    try:
        parallel_status = get_parallel_processing_status()
        processing_status = "ok" if parallel_status['parallelism_enabled'] else "degraded"
    except Exception as e:
        processing_status = f"error: {str(e)}"
        parallel_status = {}
    
    return {
        "status": "healthy" if redis_status == "ok" and processing_status == "ok" else "degraded",
        "redis": redis_status,
        "parallel_processing": processing_status,
        "system_info": parallel_status,
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
