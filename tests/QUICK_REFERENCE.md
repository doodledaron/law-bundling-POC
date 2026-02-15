# Quick Reference: Test Suite Usage

## 🚀 Quick Start

```bash
cd tests/

# Warm up containers (after docker-compose up -d)
python3 01_summarization_container_warmup_test.py

# Test summarization with single document
python3 02_summarization_single_upload_test.py

# Run stress test (optional, takes 30-120 minutes)
python3 03_summarization_stress_test_multipage.py

# Test new relevance extraction feature
python3 04_relevance_extraction_test.py
```

---

## 📋 Test Overview

| # | Name | Purpose | Duration | Feature |
|---|------|---------|----------|---------|
| 01 | Container Warmup | Initialize 5 workers | 2-3 min | Summarization |
| 02 | Single Upload | Basic API test | 5-15 min | Summarization |
| 03 | Stress Test | Performance analysis | 30-120 min | Summarization |
| 04 | Relevance | Token-efficient extraction | 10-30 min | Relevance ⭐ NEW |

---

## 🔑 Setup

### 1. Configure `.env`
```bash
# Required
GEMINI_API_KEY=your_key_here
API_KEYS=test-key-1,test-key-2,test-key-3

# Optional
API_BASE_URL=http://localhost:8000
```

### 2. Start Containers
```bash
docker-compose up -d
sleep 30  # Wait for initialization
```

### 3. Run Tests
```bash
cd tests/
python3 01_summarization_container_warmup_test.py
```

---

## ✅ Expected Results

### Test 01: Container Warmup
```
✅ Jobs completed: 4/5
✅ Upload success rate: 5/5
🎉 Warmup SUCCESS!
```

### Test 02: Single Upload
```
✅ Upload successful!
✅ Processing completed!
📋 Results: [summary text]
```

### Test 03: Stress Test
```
✅ Completed: 4/4
📊 Success Rate: 100%
Output: processing_report_YYYYMMDD_HHMMSS.xlsx
```

### Test 04: Relevance Extraction
```
✅ Relevance extraction completed
📌 Relevances extracted: 2-3
💾 Tokens: 2,500-5,000
💰 Cost: $0.0002-0.0005
```

---

## 🔍 Key Features Tested

### Summarization (Tests 01-03)
- ✅ Document upload and validation
- ✅ Multi-container processing
- ✅ PPStructure (layout analysis) + Gemini (NLP)
- ✅ Document chunking (5-page chunks)
- ✅ Parallel processing across 5 workers
- ✅ Job status polling
- ✅ Result retrieval

### Relevance Extraction (Test 04) ⭐ NEW
- ✅ Token-efficient relevance extraction
- ✅ Pinpoint references (Para [XX], page X)
- ✅ Evidence quotes (≤25 words, with sources)
- ✅ Document type classification
- ✅ Smart text slicing (800-char windows)
- ✅ Token usage tracking
- ✅ Cost estimation ($0.10/1M input, $0.40/1M output)
- ✅ Page allocation: 90% Gemini + 10% PPStructure for large docs

---

## 💡 Common Issues & Solutions

### "API not available"
```bash
docker-compose ps                    # Check status
docker-compose logs api              # View errors
docker-compose down && docker-compose up -d  # Restart
python3 01_summarization_container_warmup_test.py  # Try again
```

### "Authentication failed"
```bash
# Verify API key is set
cat .env | grep API_KEYS

# Export if missing
export API_KEYS="test-key-1,test-key-2"
```

### Timeout errors
- Increase document size limit in Docker memory settings
- Check container logs for processing bottlenecks
- For stress tests, reduce page counts

### Memory errors
```bash
# Increase Docker memory for test execution
docker update --memory 60g law-redis
```

---

## 📊 Understanding Results

### Container Warmup
- 4/5+ containers warmed = Ready for production
- <3/5 containers = Wait and retry

### Single Upload  
- Status: COMPLETED = Success
- Processing time: Baseline for performance
- Summary length: Should be 2-4 paragraphs

### Stress Test
- Success rate ≥80% = Good
- Average pages/sec = Performance metric
- Memory usage stable = No memory leaks

### Relevance Extraction
- Relevances count: 1-5 per document (typical)
- Pinpoints: At least 1 per relevance
- Evidence quotes: 2-6 per relevance
- Token efficiency: Lower tokens = better cost

---

## 🎯 Production Verification Checklist

Before deploying to production:

- [ ] Test 01: Container warmup successful (≥4/5 warmed)
- [ ] Test 02: Single document processes end-to-end
- [ ] Test 04: Relevance extraction working correctly
- [ ] Cost estimates match expected rates
- [ ] No memory leaks in stress test
- [ ] Processing times acceptable for your use case
- [ ] API response times consistent

---

## 📈 Performance Targets

| Metric | Target | Actual |
|--------|--------|--------|
| Container warmup time | <5 min | ___ |
| 5-page document | <20s | ___ |
| 50-page document | <60s | ___ |
| 100-page document | <120s | ___ |
| API response time | <2s | ___ |
| Success rate | ≥95% | ___ |

---

## 🔗 Related Files

- Main documentation: `/CLAUDE.md`
- API routes: `/routes/api_routes.py`, `/routes/relevance_routes.py`
- Processors: `/text_based_processor.py`, `/relevance_processor.py`
- Tasks: `/tasks/gemini_tasks.py`, `/tasks/relevance_tasks.py`
- Config: `/config.py`

---

## 📞 Support

For issues or questions:
1. Check test output for specific error messages
2. Review `/CLAUDE.md` for architecture details
3. Check Docker logs: `docker-compose logs [service-name]`
4. Verify `.env` configuration
5. Ensure all dependencies installed: `pip install -r requirements.txt`

---

## ✨ Recent Updates

### New Relevance Extraction Feature (Test 04)
- Token-efficient legal relevance extraction
- Smart text slicing (improved from 200→800 char windows)
- Merged overlapping keyword windows
- Cost estimation integrated
- Pinpoint references with page numbers
- Evidence quotes with source attribution

---

*Last updated: 2026-02-15*
