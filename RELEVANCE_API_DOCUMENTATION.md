# Legal Relevance Extraction API

## Base URL
```
http://204.12.246.205:8000
```

## Authentication
All requests require API key in header:
```
X-API-Key: your-api-key-here
```

---

## Endpoints

### POST /api/relevance
Upload document for relevance extraction.

**Request:**
```http
POST /api/relevance
Content-Type: multipart/form-data
X-API-Key: your-api-key-here

file: <document.pdf>
```

**Response:**
```json
{
  "job_id": "a233db00-49b0-4a9b-9086-ed4c342db01c",
  "filename": "legal_case.pdf",
  "status": "submitted",
  "polling_endpoint": "/api/relevance/a233db00-49b0-4a9b-9086-ed4c342db01c"
}
```

### GET /api/relevance/{job_id}
Check status and retrieve results.

**Request:**
```http
GET /api/relevance/{job_id}
X-API-Key: your-api-key-here
```

**Response (Processing):**
```json
{
  "status": "PROCESSING",
  "progress": 45,
  "message": "Relevance extraction in progress..."
}
```

**Response (Completed):**
```json
{
  "status": "COMPLETED",
  "progress": 100,
  "results": {
    "document_type": "case",
    "relevances": [
      {
        "relevance_text": "The Court held that limitation periods for breach of trust claims do not begin to run until the beneficiary discovers the breach.",
        "pinpoints": [
          {
            "reference": "Para [42]",
            "page": 12,
            "type": "paragraph"
          }
        ],
        "evidence_quotes": [
          {
            "text": "time does not begin to run until the beneficiary has knowledge",
            "page": 12,
            "pinpoint_ref": "Para [42]",
            "support_for": "limitation period rule"
          }
        ]
      }
    ],
    "total_pages": 50,
    "pages_processed": {
      "gemini": 45,
      "ppstructure": 5
    },
    "token_usage": {
      "input_tokens": 12500,
      "output_tokens": 450,
      "total_tokens": 12950
    },
    "estimated_cost": 0.001430
  }
}
```

---

## Response Schema

### Document Types
- `case`: Court judgments, decisions
- `statute`: Legislation, acts, regulations
- `notice`: Guidelines, circulars
- `other`: Other legal documents

### Relevance Object
```typescript
{
  relevance_text: string,        // 2-4 line legal principle
  pinpoints: [
    {
      reference: string,          // "Para [42]", "Section 5(2)"
      page: number,
      type: string               // "paragraph", "section", "page"
    }
  ],
  evidence_quotes: [
    {
      text: string,              // Verbatim quote (≤25 words)
      page: number,
      pinpoint_ref: string,      // Links to pinpoint
      support_for: string        // What this quote supports
    }
  ]
}
```

### Cost Information
- **Pricing**: $0.10 per 1M input tokens, $0.40 per 1M output tokens
- **Typical costs**:
  - 5-page doc: ~$0.0002
  - 50-page doc: ~$0.0015
  - 100-page doc: ~$0.0030

---

## Usage Example

### cURL
```bash
# Upload
curl -X POST \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf" \
  http://204.12.246.205:8000/api/relevance

# Check status
curl -H "X-API-Key: your-api-key" \
  http://204.12.246.205:8000/api/relevance/{job_id}
```

### Python
```python
import requests
import time

API_URL = "http://204.12.246.205:8000"
headers = {"X-API-Key": "your-api-key"}

# Upload
with open("document.pdf", "rb") as f:
    response = requests.post(
        f"{API_URL}/api/relevance",
        files={"file": f},
        headers=headers
    )
job_id = response.json()["job_id"]

# Poll until complete
while True:
    response = requests.get(f"{API_URL}/api/relevance/{job_id}", headers=headers)
    data = response.json()
    
    if data["status"] == "COMPLETED":
        relevances = data["results"]["relevances"]
        break
    elif data["status"] == "FAILED":
        raise Exception(data.get("error"))
    
    time.sleep(15)
```


---

## HTTP Status Codes
- `200`: Success
- `400`: Invalid file format
- `401`: Missing/invalid API key
- `404`: Job ID not found
- `500`: Server error

---

## Key Differences from Summarization API

| Feature | Summarization (`/api/upload`) | Relevance (`/api/relevance`) |
|---------|-------------------------------|------------------------------|
| Output | Full document summary | Focused legal principles |


---

## Notes
- Supported formats: PDF, JPEG, PNG
- No file size limit
- Results include precise pinpoint references for legal research
- Evidence quotes are verbatim extracts (max 25 words each)
- Cost tracking included in every response

---

*API Version: 2.0.0*  

