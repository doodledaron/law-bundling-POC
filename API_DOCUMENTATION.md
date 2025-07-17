# Law Document Processing API - Integration Guide

## Base URL
```
http://204.12.246.205:8000/
```

## Authentication
Currently no authentication required.

---

## API Endpoints

### 1. Document Upload
**Upload a document for processing**

```http
POST /api/upload
Content-Type: multipart/form-data
```

**Request Parameters:**
- `file` (required): Document file (PDF, JPEG, PNG)

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
```

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

1. **Upload Document**: `POST /api/upload` with file
2. **Get Job ID**: Extract `job_id` from upload response
3. **Poll Status**: `GET /api/job/{job_id}` periodically until status is `COMPLETED` or `FAILED`
4. **Process Results**: Use the `results` object from completed response

## Error Handling

- **400**: Invalid file format or empty file
- **404**: Job ID not found
- **500**: Processing system error

Poll every 15-30 seconds until completion. 