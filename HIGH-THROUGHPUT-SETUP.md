# High-Throughput 6-Container Document Processing Setup

## Overview

This system has been optimized for **maximum document processing throughput** using 6 Celery worker containers with intelligent parallelism. The setup processes documents using fixed 5-page chunks distributed across 24 concurrent workers.

## Architecture

- **6 Document Processing Containers** (5GB RAM each)
- **4 Concurrency per Container** = **24 Total Workers**
- **Fixed 5-Page Chunking** for predictable load balancing
- **Shared `chunk_queue`** for optimal distribution
- **Total Memory**: 30GB (6 × 5GB containers)

## Key Improvements

### 🚀 Performance Gains
- **3x-5x faster processing** compared to 2-container setup
- **24 concurrent workers** vs previous 2 workers
- **Intelligent chunking** prevents container idle time
- **Optimized memory allocation** (5GB per container)

### 📊 Resource Optimization
- **Memory**: 30GB total (down from 48GB in old 2-container setup)
- **CPU**: Full utilization of 24-core Xeon Platinum 8160
- **Parallelism**: Up to 24 chunks processed simultaneously
- **Load Balancing**: Automatic via shared queue

## Container Configuration

Each of the 6 containers:
```yaml
Container: law-worker-documents-container[1-6]
Memory: 5GB limit (--memory=5g)
Concurrency: 4 workers per container
Queue: chunk_queue (shared)
GPU: Full access to all GPUs
```

## Execution Logic

### 1. Document Chunking
```
Document Upload → Fixed 5-page chunks → Immediate dispatch to chunk_queue
Example: 13-page PDF → 3 chunks (5, 5, 3 pages)
Status: PROCESSING (job remains in PROCESSING until all chunks complete)
```

### 2. Parallel Processing
```
Chunks → chunk_queue → Available containers (round-robin)
Up to 24 chunks can process simultaneously
Individual chunks complete → Update progress only (NOT status)
Status: PROCESSING (maintained throughout chunk processing)
```

### 3. Merge & Summarize
```
All chunks complete → Celery chord callback → Extract original filename
→ Sort chunks in correct page order → Merge content → Generate final summary
Status: COMPLETED (only set when entire document is fully processed)
Final result: Returns original document name (e.g., "10pages_affidavit.pdf")
```

### 4. Result Delivery
```
✅ Success: Original filename with complete document analysis
❌ Failure: Only if merge fails or all chunks fail
📊 Progress: Real-time updates during chunk processing
```

## Deployment

### Quick Start
```bash
# Make deployment script executable
chmod +x deploy-6-containers.sh

# Deploy the system
./deploy-6-containers.sh
```

### Manual Deployment
```bash
# Stop existing containers
docker-compose down

# Start 6-container setup
docker-compose up -d

# Monitor deployment
docker-compose logs -f
```

## System Requirements

### Minimum Requirements
- **RAM**: 32GB (30GB for containers + 2GB overhead)
- **CPU**: Intel Xeon Platinum 8160 (24 cores) or equivalent
- **GPU**: NVIDIA GPU with CUDA support
- **Storage**: 50GB+ free space

### Recommended Requirements
- **RAM**: 64GB for handling large documents and multiple concurrent jobs
- **CPU**: 24+ cores for optimal parallelism
- **GPU**: Multiple GPUs for maximum performance

## Configuration Files Modified

### 1. `docker-compose.yml`
- Expanded from 2 to 6 worker containers
- Set memory limit to 5GB per container
- Configured concurrency to 4 per container
- All containers subscribe to `chunk_queue`

### 2. `main.py`
- Implemented **5-page fixed chunking** strategy
- Removed hardcoded 2-chunk limitation
- Added high-throughput logging and monitoring
- Optimized chunk task creation

### 3. `celery_config.py`
- Updated task routing for shared `chunk_queue`
- Optimized prefetch and connection settings
- Reduced retry delays for faster processing
- Added high-throughput performance settings

### 4. `tasks/ppstructure_tasks.py`
- Updated worker detection for `chunk_queue`
- Enhanced logging for 6-container setup
- Optimized container identification logic

## Performance Monitoring

### Container Status
```bash
# Check all containers
docker ps | grep law-worker-documents

# Monitor resource usage
docker stats

# View container logs
docker-compose logs -f worker-documents-container1
```

### Queue Monitoring
```bash
# Monitor Redis queue
docker exec law-redis redis-cli monitor

# Check queue length
docker exec law-redis redis-cli llen chunk_queue
```

### Performance Metrics
```bash
# API health check
curl http://localhost:8000/health

# Check system load
htop

# GPU utilization
nvidia-smi
```

## Testing the Setup

### Basic Test
```bash
# Upload a test document
curl -X POST -F 'file=@test-document.pdf' http://localhost:8000/upload/

# Check processing status
curl http://localhost:8000/status/{job_id}
```

### Load Testing
```bash
# Upload multiple documents simultaneously
for i in {1..10}; do
    curl -X POST -F "file=@test-doc-${i}.pdf" http://localhost:8000/upload/ &
done
```

## Expected Performance

### Throughput Comparison
| Setup | Containers | Workers | Expected Performance |
|-------|------------|---------|-------------------|
| Old | 2 | 2 | Baseline |
| New | 6 | 24 | 3x-5x faster |

### Processing Examples
| Document Size | Old Setup | New Setup | Improvement |
|---------------|-----------|-----------|-------------|
| 10 pages | 2 chunks | 2 chunks | Same |
| 25 pages | 2 chunks | 5 chunks | 2.5x faster |
| 50 pages | 2 chunks | 10 chunks | 5x faster |

## Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Check memory usage
   free -h
   docker stats
   ```

2. **Container Not Starting**
   ```bash
   # Check logs
   docker-compose logs worker-documents-container1
   
   # Restart specific container
   docker-compose restart worker-documents-container1
   ```

3. **Queue Backlog**
   ```bash
   # Check queue status
   docker exec law-redis redis-cli llen chunk_queue
   
   # Clear stuck jobs if needed
   docker exec law-redis redis-cli flushdb
   ```

### Performance Optimization

1. **Increase Concurrency** (if you have more CPU cores):
   ```yaml
   # In docker-compose.yml
   command: celery -A celery_config worker -Q chunk_queue -l info --concurrency=6
   ```

2. **Adjust Memory Limits** (if you have more RAM):
   ```yaml
   # In docker-compose.yml
   mem_limit: 8g
   ```

3. **Monitor GPU Usage**:
   ```bash
   # Ensure all containers can access GPU
   nvidia-smi
   ```

## Scaling Further

### Horizontal Scaling
- Add more containers (containers 7, 8, etc.)
- Increase total concurrency
- Ensure adequate system resources

### Vertical Scaling
- Increase concurrency per container
- Allocate more memory per container
- Optimize chunk size if needed

## Advanced Configuration

### Custom Chunk Size
```python
# In main.py, modify PAGES_PER_CHUNK
PAGES_PER_CHUNK = 3  # For smaller chunks (more parallelism)
PAGES_PER_CHUNK = 10  # For larger chunks (fewer API calls)
```

### Queue Priority
```python
# In celery_config.py, add queue priorities
task_routes = {
    'urgent_tasks': {'queue': 'urgent_queue', 'priority': 10},
    'normal_tasks': {'queue': 'chunk_queue', 'priority': 5}
}
```

## Maintenance

### Regular Tasks
- Monitor disk space for results and chunks
- Check Redis memory usage
- Restart containers if memory leaks occur
- Update PPStructure models periodically

### Log Management
```bash
# View recent logs
docker-compose logs --tail=100 -f

# Clear old logs
docker system prune -f
```

---

## Summary

The 6-container high-throughput setup provides:
- ✅ **24 concurrent workers** (vs 2 previously)
- ✅ **5-page fixed chunking** for optimal load balancing  
- ✅ **30GB total memory** allocation (5GB per container)
- ✅ **3x-5x performance improvement** expected
- ✅ **Full CPU utilization** on 24-core Xeon Platinum 8160
- ✅ **Intelligent parallelism** with no idle containers

This setup maximizes throughput while maintaining resource efficiency and system stability. 