# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Service Overview

This is a production-grade legal document processing service that provides OCR, text extraction, and document analysis via FastAPI. The service uses a hybrid approach combining PaddleOCR's PPStructure for layout analysis and Google's Gemini API for intelligent text processing.

## Architecture

### Core Components
- **FastAPI Application** (`main.py`): HTTP API server handling uploads and job management
- **Celery Workers** (`celery_config.py`): Distributed task processing with Redis as broker
- **PPStructure Tasks** (`tasks/ppstructure_tasks.py`): Heavy OCR and layout analysis processing
- **Text Processor** (`text_based_processor.py`): Gemini API integration for text analysis
- **Docker Deployment**: 5-container setup for high-throughput parallel processing

### Processing Strategy
- **Small documents (≤6 pages)**: Gemini-only processing for speed
- **Large documents (≥7 pages)**: Mixed processing - 30% PPStructure + 70% Gemini
- **Chunking**: Fixed 5-page chunks for optimal load balancing
- **Parallel Processing**: Up to 15 concurrent chunk processors across 5 containers

### Key Services
- `api`: FastAPI web server (port 8000)
- `redis`: Message broker and result backend (port 6379)
- `worker-documents-container1-5`: Processing workers with GPU support
- `flower`: Celery monitoring dashboard (port 5555)

## Development Commands

### Environment Setup
```bash
# Copy environment template and configure
cp .env.example .env
# Edit .env with required configuration:
# - GEMINI_API_KEY=your_gemini_api_key
# - API_KEYS=api-key-1,api-key-2,api-key-3
# - REDIS_URL=redis://localhost:6379/0

# Install dependencies
pip install -r requirements.txt
```

### Running the Service

#### Production Deployment (Recommended)
```bash
# Deploy full 5-container high-throughput setup
chmod +x deploy-5-containers.sh
./deploy-5-containers.sh

# Check deployment status
docker-compose ps
docker-compose logs -f api
```

#### Development Mode
```bash
# Start with Docker Compose
docker-compose up -d

# Or run API locally for development
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing Commands

#### Health Check
```bash
curl http://localhost:8000/health
```

#### API Testing
```bash
# Simple upload test (requires API key)
curl -H "X-API-Key: your-api-key" -X POST -F "file=@sample.pdf" http://localhost:8000/api/upload

# Development interface test (no auth required)
curl -X POST -F "file=@sample.pdf" http://localhost:8000/dev/upload

# Run comprehensive test suite
python3 simple_api_test.py

# Container warmup test (after deployment)
python3 warmup_test.py
```

#### System Diagnostics
```bash
# System health check
python diagnose_system.py

# Memory monitoring
python check_memory.py

# Restart system if needed
python restart_system.py
```

### Monitoring Commands

#### Container Status
```bash
# Check all containers
docker-compose ps

# View logs
docker-compose logs -f worker-documents-container1
docker-compose logs -f api

# Monitor resource usage
docker stats
```

#### Redis Queue Monitoring
```bash
# Monitor Redis commands
docker exec law-redis redis-cli monitor

# Check queue status
docker exec law-redis redis-cli INFO
```

#### Celery Monitoring
```bash
# Access Flower dashboard
open http://localhost:5555

# Check worker status
celery -A celery_config inspect active
```

## Important Configuration

### Environment Variables
- `GEMINI_API_KEY`: Required for AI processing
- `REDIS_URL`: Message broker connection (default: redis://localhost:6379/0)
- `API_KEYS`: Comma-separated list of valid API keys for authentication (required for production)
- `MAX_FILE_SIZE`: Upload limit in bytes (disabled: no size limit)
- `PAGES_PER_CHUNK`: Chunking strategy (default: 5 pages)

### Memory Configuration
- Each worker container: 9GB RAM limit
- Redis: 24GB memory allocation
- Total system requirement: 32GB+ RAM recommended (69GB allocation)
- Shared volumes: `uploads_data`, `results_data`, `redis-data`

### GPU Configuration
- All 5 worker containers require NVIDIA GPU access
- CUDA environment variables configured for stability
- TensorRT optimizations enabled for performance

## Processing Flow

1. **Upload**: Document uploaded via `/api/upload` or `/upload`
2. **Job Creation**: Unique job ID generated, stored in Redis
3. **Document Analysis**: Page count determines processing strategy
4. **Chunking**: Large documents split into 5-page chunks
5. **Parallel Processing**: Chunks distributed across 5 worker containers
6. **Results Merging**: Text combined and summary generated
7. **Storage**: Results saved to `/results/{job_id}/` directory

## API Endpoints

### Production Endpoints (Authentication Required)
- `POST /api/upload`: Single document upload (returns job_id) - **Requires X-API-Key header**
- `GET /api/job/{job_id}`: Check processing status - **Requires X-API-Key header**
- `GET /health`: System health check (public, no auth required)

### Development Endpoints (No Authentication)
- `GET /dev/`: Web upload interface
- `GET /dev/bulk`: Bulk processing interface
- `POST /dev/bulk-upload`: Multiple document upload
- `POST /dev/upload`: Single upload (HTML form)
- `GET /dev/job/{job_id}`: Job status page
- `GET /dev/api/results`: List all processed documents
- `GET /dev/api/parallel/status`: Processing system status
- `GET /dev/api/containers/status`: Container status

## File Structure

### Key Directories
- `uploads/`: Temporary file storage for uploaded documents
- `results/`: Processed document results (organized by job_id)
- `chunks/`: Temporary chunk storage during processing
- `tasks/`: Celery task definitions
- `templates/`: HTML templates for web interface

### Configuration Files
- `docker-compose.yml`: Full 5-container deployment
- `celery_config.py`: Celery worker and queue configuration
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container image definition

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce `MAX_DOCUMENT_WORKERS` or container memory limits
2. **Redis Connection**: Check `REDIS_URL` and ensure Redis container is running
   - For local debugging scripts (`diagnose_system.py`, `restart_system.py`), uncomment Redis port in `docker-compose.yml`
3. **GPU Access**: Verify NVIDIA Docker runtime and GPU drivers
4. **Upload Failures**: Check format support (PDF/JPEG/PNG only)

### Log Analysis
- API errors: `docker-compose logs api`
- Worker errors: `docker-compose logs worker-documents-container1`
- Redis issues: `docker-compose logs redis`
- System diagnostics: `python diagnose_system.py` (requires Redis port exposed)

### Performance Optimization
- Monitor container resource usage with `docker stats`
- Use warmup script after deployment: `python3 warmup_test.py`
- Adjust chunk size via `PAGES_PER_CHUNK` environment variable
- Redis memory optimized to 24GB for large document queues
- Optimized 5-container setup reduces memory footprint while maintaining performance

## Security Notes

### Authentication
- **API key authentication** implemented for production endpoints (`/api/*`)
- **Development endpoints** (`/dev/*`) require no authentication for internal use
- **Environment-based configuration**: API keys managed via `API_KEYS` environment variable
- **Backward compatibility**: Authentication can be disabled by not setting `API_KEYS`

### Data Security
- Files are temporarily stored and cleaned up after processing
- Redis backend stores job status and results temporarily
- All file uploads are validated for type (no size limits)
- Container isolation provides security boundaries between workers
- **Redis port not exposed externally** - only accessible within Docker network (production-safe)

### Usage Examples
```bash
# Production API usage (with authentication)
curl -H "X-API-Key: your-api-key" \
     -F "file=@document.pdf" \
     http://localhost:8000/api/upload

# Development interface usage (no authentication)
curl -F "file=@document.pdf" \
     http://localhost:8000/dev/upload
```