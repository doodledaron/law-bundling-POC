# Law Document Bundling POC

Automated legal document processing pipeline using hybrid OCR (PaddlePaddle PPStructure) and AI text extraction (Google Gemini). Upload a PDF → get structured text, summaries, key dates, and party information.

## Architecture

```
┌─────────┐     ┌───────┐     ┌──────────────────────────────────┐
│ FastAPI  │────▶│ Redis │────▶│ Celery Workers (x5)              │
│ :8000   │     │ :6379 │     │  ├─ PPStructure OCR (GPU)        │
└─────────┘     └───────┘     │  ├─ Gemini AI text extraction    │
                              │  └─ Merge & Summarize            │
                              └──────────────────────────────────┘
                              ┌──────────────────────────────────┐
                              │ Supporting Services               │
                              │  ├─ Celery Beat (scheduled tasks)│
                              │  ├─ Maintenance Worker (cleanup) │
                              │  └─ Flower Dashboard (:5555)     │
                              └──────────────────────────────────┘
```

## Quick Start

```bash
# Build and start everything
make start

# Or manually
docker compose build
docker compose up -d

# Check status
make status

# View logs
make logs
```

## Requirements

- Docker & Docker Compose
- NVIDIA GPU + drivers (for PPStructure OCR workers)
- `.env` file with:
  ```
  GEMINI_API_KEY=your-gemini-api-key
  API_KEYS=your-api-key-for-auth
  ```

## Make Commands

Run `make help` to see all available commands:

| Command | Description |
|---------|-------------|
| `make start` | Build and start all services |
| `make up` | Start services (no rebuild) |
| `make down` | Stop all services |
| `make restart` | Stop + start |
| `make rebuild` | Full rebuild (no cache) + start |
| `make logs` | Follow all container logs |
| `make logs-api` | API logs only |
| `make logs-workers` | All worker logs |
| `make logs-worker N=2` | Specific worker (1-5) |
| `make status` | Show running containers |
| `make health` | API health check |
| `make disk` | Check data directory sizes |
| `make clean` | Delete old files (7d results, 1d chunks) |
| `make test` | Run test document upload |
| `make shell` | Bash into worker container |
| `make celery-active` | Show active Celery tasks |

## API Usage

All endpoints require the `X-API-Key` header.

### Upload a document
```bash
curl -X POST \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf" \
  http://localhost:8000/api/upload
```

### Check job status
```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/api/job/{job_id}
```

### Health check
```bash
curl http://localhost:8000/health
```

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for full API reference.

## Project Structure

```
├── main.py                    # FastAPI app entrypoint
├── celery_config.py           # Celery configuration & task routing
├── text_based_processor.py    # Gemini AI text extraction (with retry logic)
├── config.py                  # App configuration
├── tasks/
│   ├── __init__.py            # Task exports
│   ├── ppstructure_tasks.py   # PPStructure OCR processing
│   ├── gemini_tasks.py        # Gemini-only document processing
│   ├── merge_tasks.py         # Chunk merging & summarization
│   ├── helpers.py             # Shared utilities (bounding boxes, etc.)
│   ├── utils.py               # Job status & progress tracking
│   └── maintenance.py         # Cleanup & system stats tasks
├── routes/                    # FastAPI route handlers
├── services/                  # Business logic (auth, processing)
├── templates/                 # HTML templates
├── Dockerfile                 # Main worker image (PaddlePaddle + GPU)
├── Dockerfile.lite            # Lightweight image (API, beat, flower)
├── docker-compose.yml         # Full stack (5 workers)
└── Makefile                   # Dev/ops commands
```

## Services

| Service | Container | Port | Image |
|---------|-----------|------|-------|
| API (FastAPI) | `law-api` | 8000 | Dockerfile.lite |
| Redis | `law-redis` | 6379 | redis:7-alpine |
| Worker 1-5 | `law-worker-documents-*` | - | Dockerfile (GPU) |
| Maintenance | `law-worker-maintenance` | - | Dockerfile.lite |
| Beat | `law-beat` | - | Dockerfile.lite |
| Flower | `law-flower` | 5555 | Dockerfile.lite |

## Data Lifecycle

| Directory | Contents | Cleanup |
|-----------|----------|---------|
| `uploads/` | Uploaded PDFs | Auto-deleted after 7 days |
| `results/` | Extracted text, images, JSON | Auto-deleted after 7 days |
| `chunks/` | Temporary split PDFs | Auto-deleted after 1 day |
| Redis | Job metadata (~2KB/job) | Keys expire after 7 days |

Cleanup runs daily via Celery Beat (`cleanup_expired_results` task).