# Law Document Processing API - Integration Guide

## Base URL
```
http://204.12.246.205:8000/
```

## Authentication
**Authentication is required for all API endpoints using API Key.**

### API Key Header
All requests must include a valid API key in the request header:
```
X-API-Key: your-api-key-here
```

### Getting an API Key
The API key will be provided separately for security. Include it in all requests to `/api/*` endpoints.

### Authentication Errors
- **401 Unauthorized**: Missing or invalid API key
- **Error Response Format**:
```json
{
  "detail": {
    "error": "Missing API key",
    "message": "Please provide a valid API key in the 'X-API-Key' header",
    "required_header": "X-API-Key"
  }
}
```

---

## API Endpoints

### 1. Document Upload
**Upload a document for processing**

```http
POST /api/upload
Content-Type: multipart/form-data
X-API-Key: your-api-key-here
```

**Request Parameters:**
- `file` (required): Document file (PDF, JPEG, PNG)

**Headers:**
- `X-API-Key` (required): Valid API key for authentication

**Response:**
```json
{
  "job_id": "ab13b87f-2b04-49b0-8eb0-5f461605934f",
  "filename": "document.pdf",
  "status": "submitted",
  "message": "Document submitted for processing",
  "polling_endpoint": "/api/job/ab13b87f-2b04-49b0-8eb0-5f461605934f"
}
```

### 2. Job Status and Results
**Check processing status and get results**

```http
GET /api/job/{job_id}
X-API-Key: your-api-key-here
```

**Headers:**
- `X-API-Key` (required): Valid API key for authentication

**Response (Processing):**
```json
{
  "status": "PROCESSING",
  "progress": 45,
  "stage": "Processing chunk 2/3 (parallel batches)",
  "message": "Document processing in progress..."
}
```

**Response (Completed):**
```json
{
  "status": "COMPLETED",
  "progress": 100,
  "results": {
    "date": "2024-01-15",
    "summary": "Legal contract between Company A and Company B for software development services",
    "extracted_info": {
      "key_dates": "2024-01-15, 2024-06-15",
      "main_parties": "Company A, Company B",
      "case_reference_numbers": "CONTRACT-2024-001"
    },
    "total_pages": 5,
    "processing_completed_at": "2025-07-16T15:30:55.646878"
  }
}
```

**Response (Failed):**
```json
{
  "status": "FAILED",
  "error": "Document processing failed: Invalid PDF format",
  "message": "Processing failed due to file format issues"
}
```

### 3. Health Check
**Check API availability**

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "redis": "ok",
  "parallel_processing": "ok",
  "version": "2.0.0"
}
```

---

## Integration Flow

1. **Obtain API Key**: Get a valid API key from your system administrator
2. **Upload Document**: `POST /api/upload` with file and API key in header
3. **Get Job ID**: Extract `job_id` from upload response
4. **Poll Status**: `GET /api/job/{job_id}` with API key until status is `COMPLETED` or `FAILED`
5. **Process Results**: Use the `results` object from completed response

## Example Integration

### cURL Example
```bash
# Upload document
curl -X POST \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@document.pdf" \
  http://204.12.246.205:8000/api/upload

# Check status
curl -H "X-API-Key: your-api-key-here" \
  http://204.12.246.205:8000/api/job/ab13b87f-2b04-49b0-8eb0-5f461605934f
```


### HTTP Status Codes
- **200**: Success
- **400**: Invalid file format or empty file
- **401**: Missing or invalid API key
- **404**: Job ID not found
- **500**: Processing system error


## Polling Recommendations
- **Poll interval**: 15-30 seconds
- **Timeout**: 10-15 minutes for large documents
- **Status progression**: PENDING → PROCESSING → COMPLETED/FAILED
