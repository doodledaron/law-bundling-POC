# Test Suite for Law Document Processing System

This directory contains all integration and functional tests for the law document processing system. Tests are organized by feature and numbered for sequential execution.

## Test Files Organization

### Summarization Pipeline Tests (Document Summarization)

These tests verify the original document summarization feature using full document processing with mixed PPStructure and Gemini integration.

#### `01_summarization_container_warmup_test.py`
- **Purpose**: Warm up all 5 worker containers after Docker Compose deployment
- **What it tests**: Container initialization, basic upload/processing workflow
- **Key features**:
  - Multi-container activation (5 worker containers)
  - Variable document sizes (2-12 pages)
  - Parallel processing verification
  - Quick health checks
- **Usage**: `python3 01_summarization_container_warmup_test.py`
- **Expected duration**: ~2-3 minutes
- **Success criteria**: At least 4/5 containers warmed up successfully

#### `02_summarization_single_upload_test.py`
- **Purpose**: Test the summarization API with single document upload
- **What it tests**:
  - Single document upload endpoint (`/api/upload`)
  - Job polling and status tracking
  - Full document summarization output
  - Results retrieval and parsing
- **Key features**:
  - Minimal PDF generation
  - Real API integration
  - Detailed result inspection
  - Error handling and recovery
- **Usage**: `python3 02_summarization_single_upload_test.py`
- **Expected duration**: ~5-15 minutes (document size dependent)
- **Success criteria**: Document processes successfully, summary generated

#### `03_summarization_stress_test_multipage.py`
- **Purpose**: Comprehensive stress test with varying document sizes
- **What it tests**:
  - Processing pipeline scalability (5-1000 pages)
  - Memory management and monitoring
  - Processing method detection (Gemini-only vs Hybrid)
  - Chunking strategy verification
  - Cost estimation accuracy
- **Key features**:
  - Dynamic PDF generation with ReportLab support
  - Memory usage tracking
  - Excel report generation
  - Detailed timing analysis
  - Parallel job monitoring capability
- **Usage**: `python3 03_summarization_stress_test_multipage.py`
- **Expected duration**: ~30-120 minutes (varies by page count)
- **Success criteria**: 80%+ success rate across all page ranges
- **Output files**:
  - `pdf_length_analysis_YYYYMMDD_HHMMSS.json` - Detailed results
  - `processing_report_YYYYMMDD_HHMMSS.xlsx` - Excel report with charts

---

### Relevance Extraction Tests (New Feature)

Tests for the token-efficient legal relevance extraction feature. This generates concise, traceable relevance summaries optimized for minimal token usage.

#### `04_relevance_extraction_test.py`
- **Purpose**: Test the relevance extraction API with token-efficient processing
- **What it tests**:
  - Relevance extraction endpoint (`/api/relevance`)
  - Token usage tracking and cost estimation
  - Pinpoint references and evidence quotes
  - Document type classification
  - Smart text slicing for large documents
  - Processing method allocation (90% Gemini / 10% PPStructure for large docs)
- **Key features**:
  - Legal document generation (cases, statutes, notices)
  - Relevance output validation
  - Cost analysis and reporting
  - Evidence quote verification
  - Processing efficiency analysis
- **Usage**: `python3 04_relevance_extraction_test.py`
- **Expected duration**: ~10-30 minutes
- **Success criteria**: 
  - All relevances extracted successfully
  - Token usage accurately tracked
  - Cost estimation provided
  - Pinpoints and evidence quotes present
  - 75%+ success rate

---

## Configuration

All tests use environment variables loaded from `.env` file:

```bash
# Required
GEMINI_API_KEY=your_api_key_here
API_KEYS=test-key-1,test-key-2,test-key-3

# Optional
API_BASE_URL=http://localhost:8000  # Default: http://localhost:8000
REDIS_URL=redis://localhost:6379/0
```

### API Key Configuration

- Tests read the first API key from the `API_KEYS` comma-separated list
- Authentication header: `X-API-Key`
- Used for both summarization and relevance extraction endpoints

---

## Running Tests

### Single Test
```bash
cd tests/
python3 01_summarization_container_warmup_test.py
```

### Sequential Test Suite (Recommended)
```bash
cd tests/
bash run_all_tests.sh
```

### Custom Test Selection
```bash
cd tests/
python3 01_summarization_container_warmup_test.py
python3 02_summarization_single_upload_test.py
python3 04_relevance_extraction_test.py
```

---

## Test Report Interpretation

### Container Warmup Test Results
```
✅ Jobs completed: 4/5
✅ Upload success rate: 5/5
```
- Indicates successful container initialization
- Safe to proceed with production workloads

### Single Upload Test Results
```
✅ Processing completed!
📋 Results:
   📄 Summary: [extracted document summary]
   📊 Total pages: 50
   ✅ Completed at: 2026-02-15T10:30:45
```
- Verify summary content is relevant
- Check processing time is reasonable

### Stress Test Results (Excel Report)
```
📊 Processing Report: processing_report_20260215_103045.xlsx
```
Columns include:
- Document, Pages, Upload Time, Total Processing Time
- Processing Method, PPStructure Chunks, Gemini Chunks
- Status, Error, Pages/Second
- Start/End Times

Key metrics:
- **Success Rate**: Should be ≥80%
- **Pages/Second**: Varies by document type and system load
- **Processing Method**: Verify correct allocation (Gemini vs Hybrid)

### Relevance Extraction Test Results
```
✅ Relevance extraction completed in 15.23s
   📋 Document type: case
   📊 Number of relevances: 3
   📌 Relevance #1:
      Text: "The Court held that the defendant is liable..."
      Pinpoints: 2
      Evidence quotes: 3
   💰 Token Usage & Cost:
      Input tokens: 2,500
      Output tokens: 300
      Total tokens: 2,800
      Estimated cost: $0.000325
```

---

## Troubleshooting

### "API not available" / Connection refused
```bash
# Check if containers are running
docker-compose ps

# Start containers
docker-compose up -d

# Wait 30-60 seconds for initialization
python3 01_summarization_container_warmup_test.py
```

### Authentication failures
```bash
# Verify .env file has API_KEYS set
cat .env | grep API_KEYS

# Ensure API key is in comma-separated list
export API_KEYS="key1,key2,key3"
```

### Memory errors during stress tests
```bash
# Reduce test page counts in the script
# Or increase system memory allocation to Docker
docker update --memory 60g law-redis
```

### Timeout during processing
- For stress tests with 500+ pages, increase `max_wait_minutes` parameter
- Check logs: `docker-compose logs worker-documents-container1`

---

## Cost Tracking

Both summarization and relevance extraction tests track API costs:

```
Input tokens:    $0.10 per 1M tokens
Output tokens:   $0.40 per 1M tokens
Estimated cost:  (input_tokens / 1M) × $0.10 + (output_tokens / 1M) × $0.40
```

### Monthly Cost Estimation Example

For relevance extraction (token-efficient):
- Average document: 10K input tokens, 300 output tokens
- Cost per document: $0.000325
- 1000 documents/month: ~$0.33/month

For full summarization (more tokens):
- Average document: 50K input tokens, 1K output tokens
- Cost per document: $0.00065
- 1000 documents/month: ~$0.65/month

---

## Test Maintenance

When adding new features:

1. Create new test file following naming convention: `XX_feature_name_test.py`
2. Number sequentially (01, 02, 03, etc.)
3. Include docstring explaining purpose and scope
4. Add to this README under appropriate section
5. Update any CI/CD pipelines to include new test

---

## CI/CD Integration

For GitHub Actions or other CI systems:

```yaml
- name: Run API tests
  run: |
    cd tests/
    python3 01_summarization_container_warmup_test.py
    python3 02_summarization_single_upload_test.py
    python3 04_relevance_extraction_test.py
```

---

## Performance Benchmarks

Expected performance metrics (production environment):

| Document Size | Processing Time | Method | Cost/Doc |
|---------------|-----------------|--------|----------|
| 5 pages       | 10-15s         | Gemini | $0.0001  |
| 10 pages      | 15-20s         | Gemini | $0.0002  |
| 50 pages      | 30-45s         | Mixed  | $0.0005  |
| 100 pages     | 45-90s         | Mixed  | $0.0010  |
| 500+ pages    | 4-8 min        | Mixed  | $0.0050  |

---

## Notes

- All tests use API key authentication (production-ready)
- Tests do NOT modify existing code
- Environment variables loaded from `.env` file
- Compatible with both local and Docker deployments
- Memory-efficient test PDF generation with optional ReportLab support
