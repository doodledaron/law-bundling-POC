# Test Execution Guide via Makefile

All tests now use `API_BASE_URL` from your `.env` file for flexible configuration.

## 📋 Available Make Commands

### Individual Tests

```bash
# 1. Warm up containers (2-3 minutes)
make test
# Runs: 01_summarization_container_warmup_test.py

# 2. Single document upload test (5-15 minutes)
make test-single
# Runs: 02_summarization_single_upload_test.py

# 3. Relevance extraction test (10-30 minutes) ⭐ NEW
make test-relevance
# Runs: 04_relevance_extraction_test.py

# 4. Comprehensive stress test (30-120 minutes, optional)
make test-stress
# Runs: 03_summarization_stress_test_multipage.py
```

### Test Suites

```bash
# Run all recommended tests (in order)
make test-all
# Equivalent to: make test && make test-single && make test-relevance

# Run complete test suite (including stress test)
make test-full
# Equivalent to: make test-all && make test-stress
```

---

## 🔧 Configuration via .env

All tests automatically use these `.env` variables:

```bash
# REQUIRED
GEMINI_API_KEY=your_api_key_here
API_KEYS=your-api-key-1,your-api-key-2

# Used by tests for API endpoint (recommended)
API_BASE_URL=http://localhost:8000

# Fallback option (if API_BASE_URL not set)
API_PORT=localhost:8000
```

### Priority Order
1. `API_BASE_URL` (if set, used directly)
2. `API_PORT` (if set, constructs `http://{API_PORT}`)
3. Default fallback: `http://localhost:8000`

---

## 🚀 Quick Start

### Development Environment (localhost)

```bash
# 1. Start containers
docker-compose up -d

# 2. Wait for initialization
sleep 30

# 3. Run tests
make test-all
```

### Production Environment (custom URL)

```bash
# 1. Update .env with your server
echo "API_BASE_URL=http://204.12.246.205:8000" >> .env

# 2. Run tests
make test-all
```

### Specific Server Testing

```bash
# Test against different servers without changing .env
API_BASE_URL=http://192.168.1.100:8000 make test-all
API_BASE_URL=http://production.server.com:8000 make test-all
```

---

## 📊 Test Execution Order Explanation

### Recommended Order: `make test-all`

```
1. make test
   └─ Warm up all 5 worker containers
   
2. make test-single
   └─ Validate basic summarization functionality
   
3. make test-relevance
   └─ Validate new token-efficient relevance extraction
```

**Why this order?**
- ✅ Container warmup first (activates workers)
- ✅ Quick validation after warmup (tests basic functionality)
- ✅ New feature validation (ensures reliability)
- ⏳ Stress test optional (time-intensive)

---

## 📈 Expected Results

### Test: `make test` (Container Warmup)
```
✅ Jobs completed: 4/5
✅ Upload success rate: 5/5
🎉 Warmup SUCCESS!
```

### Test: `make test-single` (Single Upload)
```
✅ Upload successful!
✅ Processing completed!
📄 Summary: [extracted text]
```

### Test: `make test-relevance` (Relevance Extraction)
```
✅ Relevance extraction completed
📌 Relevances extracted: 2-3
💰 Cost: $0.0002-0.0005
```

### Test: `make test-stress` (Stress Test)
```
📊 Success Rate: 80-95%
📊 Output: processing_report_YYYYMMDD_HHMMSS.xlsx
```

---

## 🔍 Monitoring Test Progress

### View test logs in real-time

```bash
# In one terminal, watch the logs
docker-compose logs -f api worker-documents-container1

# In another terminal, run tests
make test-all
```

### Check system health during tests

```bash
# Check container status
docker-compose ps

# Monitor resource usage
docker stats

# Check Redis queue
docker exec law-redis redis-cli DBSIZE
```

---

## 🆘 Troubleshooting

### Tests fail with "API not available"

```bash
# Check if API is running
curl -s http://localhost:8000/health | python3 -m json.tool

# If not running, start containers
docker-compose up -d

# Wait for startup
sleep 30

# Try tests again
make test
```

### "Authentication failed" errors

```bash
# Verify API_KEYS is set in .env
grep API_KEYS .env

# Should see output like:
# API_KEYS=your-api-key-1,your-api-key-2

# If missing, add it:
echo "API_KEYS=your-api-key-1,your-api-key-2" >> .env
```

### "Connection refused" on custom server

```bash
# Verify API_BASE_URL is correct
grep API_BASE_URL .env

# Test connectivity
curl -s http://your-server:8000/health

# If not responding, check:
# - Server is running
# - Port is correct
# - Firewall allows connection
```

### Tests timeout

```bash
# Increase Docker memory
docker update --memory 60g law-redis
docker update --memory 12g law-api

# Or reduce stress test page count in script
```

---

## 🎯 Common Use Cases

### Local Development Testing

```bash
# Set up for localhost
echo "API_BASE_URL=http://localhost:8000" >> .env

# Run quick tests
make test && make test-single

# Done! ~15 minutes total
```

### Production Validation

```bash
# Update server
sed -i '' 's|API_BASE_URL=.*|API_BASE_URL=http://production.server.com:8000|' .env

# Run complete test suite
make test-full

# Takes ~2-3 hours with stress test
```

### Testing Multiple Servers

```bash
# Test server 1
API_BASE_URL=http://server1:8000 make test-all

# Test server 2
API_BASE_URL=http://server2:8000 make test-all

# Test server 3
API_BASE_URL=http://server3:8000 make test-all
```

### CI/CD Pipeline

```bash
# .github/workflows/test.yml
- name: Run test suite
  run: make test-all
  env:
    API_BASE_URL: ${{ secrets.TEST_API_URL }}
```

---

## 📝 Test File Locations

All test files are organized in `/tests/` directory:

```
tests/
├── 01_summarization_container_warmup_test.py
├── 02_summarization_single_upload_test.py
├── 03_summarization_stress_test_multipage.py
├── 04_relevance_extraction_test.py ⭐ NEW
├── README.md
├── QUICK_REFERENCE.md
└── INDEX.md
```

---

## 💡 Tips & Best Practices

### 1. Always run `make test` first
- Warms up containers
- Ensures system is ready

### 2. Use `API_BASE_URL` environment variable
- Flexible for different servers
- Easy to override in CI/CD

### 3. Check logs if tests fail
```bash
docker-compose logs -f api
```

### 4. Run tests after deployments
```bash
git pull && docker-compose restart && sleep 30 && make test-all
```

### 5. Schedule stress tests during low-traffic periods
```bash
# Stress tests use significant resources
make test-stress  # Run at 2 AM, not during business hours
```

---

## 📊 Performance Targets

| Test | Duration | Success Rate |
|------|----------|--------------|
| Warmup | 2-3 min | ≥80% |
| Single | 5-15 min | ≥95% |
| Relevance | 10-30 min | ≥75% |
| Stress | 30-120 min | ≥80% |

---

## 🔄 Version History

- **Feb 15, 2026**: Tests updated to use `API_BASE_URL` from `.env`
- **Feb 15, 2026**: Makefile enhanced with comprehensive test commands
- **Feb 15, 2026**: Test folder reorganized with clear naming convention

---

*Last updated: 2026-02-15*
