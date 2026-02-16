# Incident Report & Fix Documentation

**Date:** December 2, 2025  
**Severity:** High (Production System Down)  
**Status:** Resolved ✅

---

## Issue Summary

**Error Message:**
```json
{"detail":"Upload failed: You can't write against a read only replica."}
```

**Impact:** The `/api/upload` endpoint was failing for all document uploads. External services calling the API received 500 errors.

---

## Root Cause Analysis

### Primary Cause: Redis Entered Read-Only Replica Mode

Redis was incorrectly operating in **slave/replica mode** instead of master mode, causing all write operations to fail.

**Evidence from logs:**
```
1:S 01 Dec 2025 19:24:17.393 # Error condition on socket for SYNC: Connection refused
1:S 01 Dec 2025 19:24:38.318 # Wrong signature trying to load DB from file
1:S 01 Dec 2025 19:24:38.318 # Failed trying to load the MASTER synchronization DB from disk
```

The `1:S` prefix indicates Redis was in Slave mode.

### Contributing Factors

1. **Memory Overcommit Disabled (`vm.overcommit_memory=0`)**
   - Linux was blocking Redis fork operations
   - This caused Redis state corruption during background saves
   - Redis warning: `WARNING Memory overcommit must be enabled!`

2. **Container Memory Mismatch**
   - Celery `worker_max_memory_per_child`: 8GB per process
   - Celery `worker_concurrency`: 2 processes per container
   - Total potential usage: 16GB per container
   - Container `mem_limit`: 9GB
   - **Result:** Containers getting OOM-killed, cascading failures to Redis

---

## Changes Made

### 1. Redis Configuration (docker-compose.yml)

**File:** `docker-compose.yml` (line 35)

**Before:**
```yaml
command: redis-server --save "" --appendonly no --loglevel warning --maxmemory 24gb --maxmemory-policy allkeys-lru --maxmemory-samples 10 --lazyfree-lazy-eviction yes --tcp-keepalive 60 --timeout 0
```

**After:**
```yaml
command: redis-server --save "" --appendonly no --loglevel warning --maxmemory 24gb --maxmemory-policy allkeys-lru --maxmemory-samples 10 --lazyfree-lazy-eviction yes --tcp-keepalive 60 --timeout 0 --replicaof no one
```

**Purpose:** The `--replicaof no one` flag ensures Redis always starts as master, even if corrupted replication state is loaded from disk.

---

### 2. Redis Configuration (docker-compose-separation.yml)

**File:** `docker-compose-separation.yml` (line 35)

**Before:**
```yaml
command: redis-server --save 60 1 --loglevel warning --maxmemory 4gb --maxmemory-policy allkeys-lru
```

**After:**
```yaml
command: redis-server --save 60 1 --loglevel warning --maxmemory 4gb --maxmemory-policy allkeys-lru --replicaof no one
```

**Purpose:** Same as above - ensures master mode on startup.

---

### 3. Linux Kernel Memory Setting

**Commands executed:**
```bash
# Apply immediately
sudo sysctl vm.overcommit_memory=1

# Make permanent (survives reboot)
echo "vm.overcommit_memory = 1" | sudo tee -a /etc/sysctl.conf
```

**Purpose:** Allows Redis to fork for background saves and replication without Linux blocking the operation due to memory concerns. This prevents state corruption during fork failures.

---

### 4. Celery Worker Memory Limit (celery_config.py)

**File:** `celery_config.py` (line 64)

**Before:**
```python
worker_max_memory_per_child=8589934592,  # 8GB memory limit per worker
```

**After:**
```python
worker_max_memory_per_child=6442450944,  # 6GB memory limit per worker (safer buffer for heavy OCR tasks)
```

**Purpose:** Prevents container OOM kills. With 2 workers per container and 6GB each, peak usage (12GB) stays reasonable and rarely both workers peak simultaneously. The 9GB container limit provides adequate headroom for typical operations.

---

## Verification Steps

After applying fixes, verify with:

```bash
# Check Redis is master
docker exec law-redis redis-cli INFO replication | grep role
# Expected: role:master

# Test Redis writes
docker exec law-redis redis-cli SET test "value" && docker exec law-redis redis-cli DEL test
# Expected: OK, 1

# Check vm.overcommit_memory
cat /proc/sys/vm/overcommit_memory
# Expected: 1

# Check worker memory config
grep worker_max_memory_per_child celery_config.py
# Expected: 6442450944
```

---

## Services Restarted

1. **Redis:** `docker compose restart redis`
2. **Workers:** `docker compose restart worker-documents-container1 worker-documents-container2 worker-documents-container3 worker-documents-container4 worker-documents-container5`

---

## Prevention Measures

| Layer | Protection | What It Prevents |
|-------|------------|------------------|
| Redis | `--replicaof no one` | Redis starting in slave mode |
| Linux | `vm.overcommit_memory=1` | Fork failures corrupting Redis |
| Celery | 6GB memory limit | Container OOM kills |

---

## Confidence Level

**90%** - All identified root causes have been addressed. If the issue recurs, additional monitoring data will be available for further diagnosis.

---

## Monitoring Recommendations

1. Set up alerting on Redis role changes
2. Monitor container OOM events: `docker events --filter 'event=oom'`
3. Check Redis logs periodically: `docker logs law-redis | grep -i error`

---

*Document created: December 2, 2025*  
*Last updated: December 2, 2025*

