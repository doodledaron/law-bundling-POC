# Current System Status - Law Document Processing

## 🎯 **What You Requested**

- **Single worker** with internal parallelism (not multiple workers)
- **PPStructure warmup** on worker startup
- **Fix job submission issues** (jobs not starting after upload)
- **Fix Celery control command errors**

## ✅ **What We've Implemented**

### 1. **Simplified Celery Configuration** (`celery_config.py`)

- Removed complex priority routing that was causing the "not enough values to unpack" error
- Simplified to single worker (concurrency=1) with internal ThreadPoolExecutor
- Enhanced memory management (6GB limit for single worker)
- Fixed broker transport options that were causing control command errors

### 2. **Internal Parallel Processing** (`tasks/document_tasks.py`)

- **NEW**: `_process_chunks_parallel_enhanced()` now uses `ThreadPoolExecutor` instead of Celery groups
- **NEW**: `_process_single_chunk_internal()` processes chunks within the worker process
- **Memory Management**: Aggressive garbage collection between batches
- **Batch Processing**: Max 3 chunks processed simultaneously within worker

### 3. **Simplified Docker Setup** (`docker-compose.yml`)

- **Single document worker** with 10GB memory (instead of primary/secondary workers)
- **Enhanced shared memory** (4GB) for internal coordination
- **Redis wait logic** to ensure proper startup sequence
- **PPStructure warmup** ready to be triggered

### 4. **Diagnostic Tools**

- **NEW**: `diagnose_system.py` - Comprehensive system health check
- **NEW**: `restart_system.py` - Clean restart with proper sequencing

### 5. **PPStructure Warmup** (`tasks/ppstructure_tasks.py`)

- Enhanced warmup task with proper testing and timing
- Lazy loading pipeline initialization
- Comprehensive error handling and status reporting

## ⚠️ **Current Issues to Fix**

### 1. **Job Submission Problem**

**Issue**: Jobs aren't starting after upload
**Likely Cause**: Task routing or import issues

**Quick Fix**:

```bash
# 1. Restart system cleanly
python restart_system.py

# 2. Check system health
python diagnose_system.py

# 3. Test warmup manually
docker exec law-worker-documents python -c "from tasks.ppstructure_tasks import warmup_ppstructure; print(warmup_ppstructure.delay().get())"
```

### 2. **Celery Control Error**

**Issue**: "ValueError('not enough values to unpack (expected 3, got 1)')"
**Fix**: Should be resolved by simplified celery_config.py

## 🚀 **Recommended Next Steps**

### Step 1: Clean Restart

```bash
# Stop everything
docker-compose down

# Clean restart
python restart_system.py
```

### Step 2: Verify System Health

```bash
# Run diagnostics
python diagnose_system.py
```

### Step 3: Test Job Submission

1. Upload a small PDF at http://localhost:8000
2. Check if job starts processing
3. Monitor progress at http://localhost:5555 (Flower)

### Step 4: Manual PPStructure Warmup (if needed)

```bash
# Warmup manually if automatic doesn't work
docker exec law-worker-documents python -c "
from tasks.ppstructure_tasks import warmup_ppstructure
result = warmup_ppstructure.delay()
print('Warmup result:', result.get(timeout=60))
"
```

## 🔧 **System Architecture Now**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │  Single Worker   │    │     Redis       │
│   (Port 8000)   │    │  (10GB Memory)   │    │  (Message Store)│
│                 │────│                  │────│                 │
│ Job Submission  │    │ ThreadPoolExecutor│    │ Job Coordination│
│ Status Tracking │    │ Internal Parallel │    │ Progress Track  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────── Flower Monitoring (Port 5555) ──────┘
```

**Processing Flow**:

1. **Job Submitted** → FastAPI receives upload
2. **Task Queued** → Redis stores job, triggers `process_document`
3. **Strategy Decision** → Large PDF = chunking, Small = direct
4. **Internal Parallel** → ThreadPoolExecutor processes 3 chunks at once
5. **Result Combination** → Merge chunk results + generate summary

## 📋 **Key Configuration Changes**

### Memory Allocation

- **Single Worker**: 10GB RAM (was: 8GB + 6GB split)
- **Shared Memory**: 4GB (for internal coordination)
- **Redis**: 4GB (increased from 2GB)

### Parallelism Strategy

- **Before**: Multiple Celery workers + Celery groups
- **Now**: Single worker + ThreadPoolExecutor (max 3 threads)

### Task Flow

- **Before**: `process_chunk_parallel` Celery task
- **Now**: `_process_single_chunk_internal` function call

## 🛠️ **If Issues Persist**

### Check Logs

```bash
# Worker logs
docker logs law-worker-documents --tail 50

# API logs
docker logs law-api --tail 20

# Redis logs
docker logs law-redis --tail 10
```

### Reset Everything

```bash
# Nuclear option - complete reset
docker-compose down -v
docker system prune -af
python restart_system.py
```

### Manual Test

```bash
# Test Redis connection
docker exec law-redis redis-cli ping

# Test worker registration
docker exec law-worker-documents celery -A celery_config inspect registered

# Test task submission
docker exec law-worker-documents python -c "
from tasks.document_tasks import process_document
result = process_document.delay('test', '/app/README.md', 'test.md')
print('Task submitted:', result.id)
"
```

## 🎯 **Expected Behavior After Fix**

1. **Upload PDF** → Immediate task submission confirmation
2. **Processing Starts** → Worker begins within 5-10 seconds
3. **Internal Parallel** → 3 chunks processed simultaneously within single worker
4. **Real-time Progress** → Status updates every few seconds
5. **Memory Efficient** → No memory leaks, worker stays under 10GB
6. **PPStructure Ready** → Models pre-loaded after warmup

The system is now configured for **single worker internal parallelism** as requested. The main fixes needed are ensuring clean startup and resolving any task routing issues.
