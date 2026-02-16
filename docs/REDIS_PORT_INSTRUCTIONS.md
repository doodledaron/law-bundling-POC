# 🔒 Redis Port Configuration

## Default Configuration (SECURE)

Redis port is **NOT exposed** to prevent external attacks. All Docker containers communicate via internal network.

```yaml
# docker-compose.yml (lines 33-34)
redis:
  # ports:
  #   - "6379:6379"  # COMMENTED OUT FOR SECURITY
```

## When You Need Local Redis Access

### Scripts That Require Redis Port:
- `diagnose_system.py` - System diagnostics
- `restart_system.py` - System restart utility

### How to Enable (Temporarily):

1. **Uncomment the Redis port** in `docker-compose.yml`:
   ```yaml
   redis:
     ports:
       - "6379:6379"
   ```

2. **Restart Redis**:
   ```bash
   docker-compose restart redis
   ```

3. **Run your script**:
   ```bash
   python diagnose_system.py
   # or
   python restart_system.py
   ```

4. **Re-secure Redis** (IMPORTANT):
   ```yaml
   redis:
     # ports:
     #   - "6379:6379"
   ```
   
   Then restart:
   ```bash
   docker-compose restart redis
   ```

## Why This Matters

Your logs showed **actual attack attempts** from external IPs:
```
law-redis | Possible SECURITY ATTACK detected
law-redis | Connection from 91.92.242.214:38074 aborted
```

Keeping Redis internal-only blocks these attacks completely.

## Production Containers (Always Work)

These containers use `redis://redis:6379/0` (internal Docker network):
- ✅ law-api
- ✅ law-worker-documents-container1
- ✅ law-worker-documents-container2
- ✅ law-worker-documents-container3
- ✅ law-worker-documents-container4
- ✅ law-worker-documents-container5
- ✅ law-worker-maintenance
- ✅ law-beat
- ✅ law-flower

**No port exposure needed for production!**
