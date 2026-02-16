# Document Processing API - Integration Guide

## Overview

This document processing system provides production-grade OCR, text extraction, and document analysis capabilities via a FastAPI interface. It's designed for high-throughput processing with intelligent load balancing and parallel processing.

## Quick Start

### 1. System Requirements

**Minimum Hardware:**
- 32GB RAM (recommended)
- 24+ CPU cores
- NVIDIA GPU (optional but recommended)
- 100GB+ storage

**Dependencies:**
- Docker & Docker Compose
- Redis server
- Gemini API key (for AI processing)

### 2. Environment Setup

Create a `.env` file with these required variables:

```env
# REQUIRED
GEMINI_API_KEY=your_gemini_api_key_here
REDIS_URL=redis://localhost:6379/0

# OPTIONAL (with defaults)
MAX_FILE_SIZE=  # No file size limit (disabled)
PAGES_PER_CHUNK=5
MAX_DOCUMENT_WORKERS=12
```

### 3. Quick Deployment

```bash
# Clone and setup
git clone [your-repo-url]
cd law-bundling-POC

# Configure environment
cp .env.example .env
# Edit .env with your values

# Deploy high-throughput setup
chmod +x deploy-6-containers.sh
./deploy-6-containers.sh

# Verify deployment
curl http://localhost:8000/health
```

## API Integration

### Core Endpoints

#### 1. Document Upload
```http
POST /upload
Content-Type: multipart/form-data

file: [PDF/JPEG/PNG file]
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "PENDING",
  "message": "Document uploaded successfully"
}
```

#### 2. Bulk Upload
```http
POST /bulk-upload
Content-Type: multipart/form-data

files: [Array of PDF/JPEG/PNG files]
```

#### 3. Job Status Check
```http
GET /api/job/{job_id}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "COMPLETED|PROCESSING|FAILED",
  "progress": 100,
  "filename": "document.pdf",
  "created_at": "2024-01-01T00:00:00",
  "processing_completed_at": "2024-01-01T00:05:00"
}
```

#### 4. Retrieve Results
```http
GET /api/results
```

**Response:**
```json
{
  "documents": [
    {
      "job_id": "uuid-string",
      "filename": "document.pdf",
      "combined_text": "Extracted text...",
      "summary": "Document summary...",
      "total_pages": 5,
      "document_type": "Contract",
      "key_dates": ["2024-01-01"],
      "main_parties": ["Company A", "Company B"]
    }
  ]
}
```

### Integration Examples

#### Python Integration
```python
import requests
import time

class DocumentProcessor:
    def __init__(self, api_base_url):
        self.api_base_url = api_base_url
    
    def upload_document(self, file_path):
        """Upload document for processing"""
        with open(file_path, 'rb') as file:
            response = requests.post(
                f"{self.api_base_url}/upload",
                files={'file': file}
            )
        return response.json()
    
    def wait_for_completion(self, job_id, timeout=300):
        """Wait for job completion with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = requests.get(f"{self.api_base_url}/api/job/{job_id}").json()
            if status['status'] == 'COMPLETED':
                return status
            elif status['status'] == 'FAILED':
                raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
            time.sleep(5)
        raise Exception("Job timeout")
    
    def get_results(self):
        """Get all processed documents"""
        response = requests.get(f"{self.api_base_url}/api/results")
        return response.json()

# Usage
processor = DocumentProcessor("http://your-domain.com")
result = processor.upload_document("document.pdf")
job_id = result['job_id']
final_result = processor.wait_for_completion(job_id)
```

#### JavaScript Integration
```javascript
class DocumentProcessor {
    constructor(apiBaseUrl) {
        this.apiBaseUrl = apiBaseUrl;
    }
    
    async uploadDocument(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${this.apiBaseUrl}/upload`, {
            method: 'POST',
            body: formData
        });
        return await response.json();
    }
    
    async checkJobStatus(jobId) {
        const response = await fetch(`${this.apiBaseUrl}/api/job/${jobId}`);
        return await response.json();
    }
    
    async waitForCompletion(jobId, timeout = 300000) {
        const startTime = Date.now();
        while (Date.now() - startTime < timeout) {
            const status = await this.checkJobStatus(jobId);
            if (status.status === 'COMPLETED') return status;
            if (status.status === 'FAILED') throw new Error(status.error);
            await new Promise(resolve => setTimeout(resolve, 5000));
        }
        throw new Error('Job timeout');
    }
}
```

## Deployment Options

### Option 1: Docker Compose (Recommended)
```bash
# High-throughput setup (6 containers)
docker-compose up -d

# Lightweight setup (1 container)
docker-compose -f docker-compose-separation.yml up -d
```

### Option 2: Kubernetes
```yaml
# Example Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: law-doc-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: law-doc-api
  template:
    metadata:
      labels:
        app: law-doc-api
    spec:
      containers:
      - name: api
        image: your-registry/law-doc-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: gemini-secret
              key: api-key
```

### Option 3: Cloud Deployment (Render/Heroku)
```yaml
# render.yaml
services:
  - type: web
    name: law-doc-api
    runtime: docker
    region: singapore
    plan: starter
    healthCheckPath: /health
    envVars:
      - key: REDIS_URL
        fromService:
          type: redis
          name: law-redis
      - key: GEMINI_API_KEY
        sync: false
```

## Performance Optimization

### System Architecture
- **6 Worker Containers**: Each with 5GB RAM
- **24 Concurrent Workers**: 4 per container
- **Fixed 5-Page Chunking**: Optimal load balancing
- **Shared Queue**: Efficient job distribution

### Performance Monitoring
```bash
# Monitor container performance
docker stats

# Check Redis queue
docker exec law-redis redis-cli monitor

# View processing logs
docker-compose logs -f worker-documents-container1
```

### Scaling Recommendations
- **Small Scale**: 1-2 containers, 2-4 workers
- **Medium Scale**: 3-4 containers, 12-16 workers
- **Large Scale**: 6+ containers, 24+ workers

## Security Considerations

### Authentication
Currently supports open access. For production:
1. Implement API key authentication
2. Add rate limiting
3. Configure CORS properly
4. Use HTTPS only

### File Security
- Files are temporarily stored in `/uploads`
- Results stored in `/results`
- Automatic cleanup after processing
- No permanent file storage

## Monitoring & Troubleshooting

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# System diagnostics
python diagnose_system.py

# Container status
docker-compose ps
```

### Common Issues
1. **Out of Memory**: Reduce `MAX_DOCUMENT_WORKERS`
2. **Redis Connection**: Check `REDIS_URL` configuration
3. **GPU Access**: Ensure Docker GPU support
4. **File Format Issues**: Check supported formats (PDF/JPEG/PNG)

### Log Monitoring
```bash
# API logs
docker-compose logs -f api

# Worker logs
docker-compose logs -f worker-documents-container1

# Redis logs
docker-compose logs -f redis
```

## Support Files

### Required Files for Integration
1. `docker-compose.yml` - Main deployment configuration
2. `requirements.txt` - Python dependencies
3. `Dockerfile` - Container image definition
4. `deploy-6-containers.sh` - Deployment script
5. `diagnose_system.py` - System diagnostics

### Optional Files
1. `restart_system.py` - System restart utility
2. `check_memory.py` - Memory monitoring
3. `HIGH-THROUGHPUT-SETUP.md` - Detailed architecture guide

## Next Steps

1. **Test Integration**: Start with single document upload
2. **Monitor Performance**: Use provided monitoring tools
3. **Scale Gradually**: Begin with 2 containers, increase as needed
4. **Implement Security**: Add authentication for production
5. **Setup Monitoring**: Configure logging and alerting

For questions or support, refer to the diagnostic tools and monitoring endpoints provided in the system. 