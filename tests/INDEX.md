# Test Suite Index

Welcome to the Law Document Processing System Test Suite! 

This folder contains all integration and functional tests organized by feature and execution order.

## 📚 Quick Navigation

### Get Started
- **New to testing?** Start with [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 5-minute overview
- **Want details?** Read [README.md](README.md) - Comprehensive documentation
- **Need the big picture?** See [TEST_ORGANIZATION_SUMMARY.txt](TEST_ORGANIZATION_SUMMARY.txt)

### Run Tests (in order)
```bash
python3 01_summarization_container_warmup_test.py      # 2-3 min
python3 02_summarization_single_upload_test.py         # 5-15 min  
python3 04_relevance_extraction_test.py                # 10-30 min (NEW!)
python3 03_summarization_stress_test_multipage.py      # 30-120 min (optional)
```

---

## 📋 Test Files

### 01_summarization_container_warmup_test.py
**Purpose:** Warm up all 5 worker containers  
**Feature:** Document Summarization (original)  
**Duration:** 2-3 minutes  
**Best for:** First-time setup verification  

### 02_summarization_single_upload_test.py
**Purpose:** Test single document upload and summarization  
**Feature:** Document Summarization (original)  
**Duration:** 5-15 minutes  
**Best for:** Quick API functionality check  

### 03_summarization_stress_test_multipage.py
**Purpose:** Comprehensive performance testing (5-1000 pages)  
**Feature:** Document Summarization (original)  
**Duration:** 30-120 minutes  
**Best for:** Performance benchmarking and stress testing  

### 04_relevance_extraction_test.py ⭐ NEW
**Purpose:** Test token-efficient legal relevance extraction  
**Feature:** Relevance Extraction (NEW)  
**Duration:** 10-30 minutes  
**Best for:** Validating the new relevance extraction feature  

---

## 🎯 Features Tested

### Summarization (Tests 01-03)
- ✅ Multi-container processing (5 workers)
- ✅ Document upload and validation
- ✅ PPStructure (layout analysis) + Gemini (NLP) integration
- ✅ Document chunking strategy
- ✅ Parallel processing
- ✅ Cost estimation

### Relevance Extraction (Test 04) ⭐ NEW
- ✅ Token-efficient relevance extraction
- ✅ Pinpoint references with page numbers
- ✅ Evidence quotes (≤25 words)
- ✅ Document type classification
- ✅ Smart text slicing (800-char windows)
- ✅ Cost tracking and estimation

---

## 🔧 Configuration

### Required (.env file)
```bash
GEMINI_API_KEY=your_key_here
API_KEYS=test-key-1,test-key-2,test-key-3
```

### Optional
```bash
API_BASE_URL=http://localhost:8000
REDIS_URL=redis://localhost:6379/0
```

---

## ✅ Success Criteria

| Test | Success Rate | Expectation |
|------|-------------|-------------|
| 01 Warmup | ≥80% | 4/5 containers warmed |
| 02 Upload | ≥95% | Single document processes |
| 03 Stress | ≥80% | Good performance across page ranges |
| 04 Relevance | ≥75% | Relevances extracted with costs |

---

## 📖 Documentation Structure

```
tests/
├── 01_summarization_container_warmup_test.py
├── 02_summarization_single_upload_test.py
├── 03_summarization_stress_test_multipage.py
├── 04_relevance_extraction_test.py (NEW)
├── README.md                          ← Detailed documentation
├── QUICK_REFERENCE.md                 ← Quick start guide
├── TEST_ORGANIZATION_SUMMARY.txt      ← Overview
└── INDEX.md                           ← This file
```

---

## 🚀 Recommended Test Flow

### For New Users
```
1. Start containers: docker-compose up -d
2. Wait 30 seconds
3. Run: python3 01_summarization_container_warmup_test.py
4. Run: python3 02_summarization_single_upload_test.py
```

### For Production Verification
```
1. Run all four tests in order
2. Verify success rates meet criteria
3. Check cost estimations match expectations
4. Review performance metrics
```

### For Continuous Integration
```
1. 01_summarization_container_warmup_test.py (mandatory)
2. 02_summarization_single_upload_test.py (mandatory)
3. 04_relevance_extraction_test.py (mandatory)
4. 03_summarization_stress_test_multipage.py (optional)
```

---

## 📊 Performance Expectations

| Document Size | Processing Time | Cost/Doc |
|---|---|---|
| 5 pages | 10-15s | $0.0001 |
| 50 pages | 30-45s | $0.0005 |
| 100 pages | 45-90s | $0.0010 |
| 500+ pages | 4-8 min | $0.0050 |

---

## 🆘 Need Help?

1. **Quick questions?** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. **Detailed info?** → [README.md](README.md)
3. **Troubleshooting?** → See "Troubleshooting" section in README.md
4. **Architecture questions?** → See `/CLAUDE.md`

---

## 📝 Notes

- All tests use API key authentication (production-ready)
- Environment variables loaded from `.env` file
- Compatible with both local and Docker deployments
- Tests do NOT modify existing code
- Original files remain in root directory (for backward compatibility)

---

## ✨ What's New

### Test 04: Relevance Extraction
- Brand new test for token-efficient legal relevance extraction
- Tests pinpoint references and evidence quotes
- Validates cost estimation accuracy
- Verifies smart text slicing with 800-character windows
- Checks page allocation strategy (90% Gemini / 10% PPStructure)

### Enhanced Organization
- Clear naming convention (NN_feature_name_test.py)
- Comprehensive documentation
- Quick reference guide
- Performance benchmarks
- Cost tracking information

---

*Last updated: 2026-02-15*  
*Test folder structure: v1.0*
