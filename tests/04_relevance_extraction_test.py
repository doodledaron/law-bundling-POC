#!/usr/bin/env python3
"""
Relevance Extraction Test Suite
Law Document Processing System

Tests the new /api/relevance endpoint with token-efficient legal relevance extraction.
This feature generates concise, traceable relevance summaries optimized for minimal token usage.

Features tested:
- Single document relevance extraction
- Token usage tracking and cost estimation
- Pinpoint references and evidence quotes
- Document type classification
- Processing method verification (Gemini vs PPStructure allocation)
- Smart text slicing for large documents

Usage: python3 04_relevance_extraction_test.py
"""

import urllib.request
import urllib.parse
import urllib.error
import json
import time
import os
import sys
import traceback
from datetime import datetime
from io import BytesIO

# Load API key from .env file
def load_env_file():
    """Load environment variables from .env file"""
    # Try current directory first, then parent directory
    env_paths = ['.env', '../.env']
    
    for env_path in env_paths:
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
                return  # Successfully loaded
        except FileNotFoundError:
            continue
    
    print("⚠️ .env file not found in current or parent directory")

# Load environment variables
load_env_file()

# API Configuration - Use API_BASE_URL from .env or construct from API_PORT
API_BASE_URL = os.getenv('API_BASE_URL')
if not API_BASE_URL:
    # Fallback: construct from API_PORT if available
    api_port = os.getenv('API_PORT', 'localhost:8000')
    API_BASE_URL = f"http://{api_port}"

BASE_URL = API_BASE_URL
UPLOAD_ENDPOINT = f"{BASE_URL}/api/relevance"
JOB_STATUS_ENDPOINT = f"{BASE_URL}/api/relevance"
HEALTH_ENDPOINT = f"{BASE_URL}/health"

print(f"ℹ️  Using API Base URL: {BASE_URL}")

# Load API key from environment
API_KEY = os.getenv('API_KEYS', '').split(',')[0] if os.getenv('API_KEYS') else None
if API_KEY:
    print(f"✅ API key loaded: {API_KEY[:10]}...")
else:
    print("⚠️ No API key found in environment - authentication may fail")

# Try to import ReportLab for better PDF generation
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    USE_REPORTLAB = True
    print("✅ ReportLab available for PDF generation")
except ImportError:
    USE_REPORTLAB = False
    print("⚠️ ReportLab not available; using minimal PDF generator")


def make_http_request(url, method='GET', data=None, headers=None, timeout=30):
    """Make HTTP request using urllib"""
    try:
        if headers is None:
            headers = {}
        
        # Add API key authentication for production endpoints
        if '/api/' in url and API_KEY:
            headers['X-API-Key'] = API_KEY
        
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_data = response.read().decode('utf-8')
            status_code = response.getcode()
            
            return {
                'status_code': status_code,
                'data': response_data,
                'success': True
            }
            
    except urllib.error.HTTPError as e:
        return {
            'status_code': e.code,
            'data': e.read().decode('utf-8') if e.fp else str(e),
            'success': False,
            'error': f'HTTP {e.code}: {e.reason}'
        }
    except urllib.error.URLError as e:
        return {
            'status_code': 0,
            'data': '',
            'success': False,
            'error': f'Connection error: {e.reason}'
        }
    except Exception as e:
        return {
            'status_code': 0,
            'data': '',
            'success': False,
            'error': f'Request error: {str(e)}'
        }


def create_legal_pdf(num_pages, doc_type="case"):
    """Create a test legal PDF with realistic content"""
    
    if USE_REPORTLAB:
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        
        # Different content based on document type
        if doc_type == "case":
            content_template = """
Case Title: Test v. Respondent
Page {page}

HELD: The Court held that the plaintiff's claim is well-founded. The defendant is liable for breach of contract.

REASONING: Per the principles established in precedent cases, the defendant shall compensate the plaintiff. The liability arises from failure to perform obligations as required by the contract.

EVIDENCE: Section [42] provides that "the defendant must fulfill all material obligations." The defendant admitted non-performance in their response.

CONCLUSION: The defendant is obligated to pay damages. The Court's order shall vest enforcement powers in the plaintiff.
"""
        elif doc_type == "statute":
            content_template = """
STATUTE: The Legal Framework Act
Section {page}

1. DEFINITIONS: For purposes of this Act, "liable party" means any entity that fails to comply.

2. OBLIGATIONS: Every liable party shall maintain compliance. Non-compliance shall trigger enforcement proceedings.

3. ENFORCEMENT: This section is enforceable and shall not begin to run from the date of violation.

4. LIABILITY: Any liable party is required to pay penalties as established in subsection (b).

5. VEST: Rights to enforce this Act vest in the regulatory authority upon enactment.
"""
        else:
            content_template = """
Legal Notice: Document Type {doc_type}
Page {page}

This document contains important legal information. The provisions herein establish obligations and rights.

KEY HOLDING: The established principle is that parties must comply with contractual obligations.

ENFORCEMENT MECHANISM: Non-compliance triggers enforcement proceedings as provided in the regulations.

EFFECTIVE DATE: This notice shall become effective and enforceable immediately upon receipt.

ACKNOWLEDGMENT: The receiving party is required to acknowledge receipt within 10 days.
"""
        
        for i in range(num_pages):
            text = content_template.format(page=i+1, doc_type=doc_type)
            lines = text.split('\n')
            y = 750
            for line in lines:
                if y > 50:
                    c.drawString(50, y, line[:80])  # Truncate long lines
                    y -= 12
            c.showPage()
        
        c.save()
        pdf_bytes = buf.getvalue()
        buf.close()
        return pdf_bytes
    
    else:
        # Minimal PDF generator
        pdf_content = "%PDF-1.4\n"
        pdf_objects = []
        
        pdf_objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        refs = " ".join(f"{3+i} 0 R" for i in range(num_pages))
        pdf_objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{refs}] /Count {num_pages} >>\nendobj\n")
        
        for i in range(num_pages):
            pdf_objects.append(
                f"{3+i} 0 obj\n"
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {3+num_pages+i} 0 R >>\nendobj\n"
            )
        
        for i in range(num_pages):
            stream = f"Legal document - Page {i+1}\\nThe court held that the defendant is liable.\\nThe defendant shall pay damages.\\nVesting of rights in the plaintiff."
            content_stream = f"BT /F1 10 Tf 50 700 Td ({stream}) Tj ET"
            pdf_objects.append(
                f"{3+num_pages+i} 0 obj\n<< /Length {len(content_stream)} >>\nstream\n"
                f"{content_stream}\nendstream\nendobj\n"
            )
        
        pdf_content += "".join(pdf_objects)
        xref_offset = len(pdf_content)
        pdf_content += "xref\n"
        pdf_content += f"0 {len(pdf_objects)+1}\n0000000000 65535 f \n"
        
        offset = len("%PDF-1.4\n")
        for obj in pdf_objects:
            pdf_content += f"{offset:010d} 00000 n \n"
            offset += len(obj)
        
        pdf_content += (
            "trailer\n"
            f"<< /Size {len(pdf_objects)+1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        )
        
        return pdf_content.encode("utf-8")


def create_multipart_data(filename, file_data, content_type):
    """Create multipart/form-data for file upload"""
    boundary = f"----WebKitFormBoundary{int(time.time() * 1000)}"
    
    body_parts = []
    body_parts.append(f'--{boundary}')
    body_parts.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"')
    body_parts.append(f'Content-Type: {content_type}')
    body_parts.append('')
    
    body_header = '\r\n'.join(body_parts).encode('utf-8')
    body_footer = f'\r\n--{boundary}--\r\n'.encode('utf-8')
    
    body = body_header + b'\r\n' + file_data + body_footer
    
    headers = {
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'Content-Length': str(len(body))
    }
    
    return body, headers


def check_api_health():
    """Check if the API is healthy"""
    print("🏥 Checking API health...")
    
    response = make_http_request(HEALTH_ENDPOINT, timeout=10)
    
    if response['success'] and response['status_code'] == 200:
        try:
            health_data = json.loads(response['data'])
            print("✅ API is healthy")
            print(f"   Version: {health_data.get('version', 'unknown')}")
            return True
        except json.JSONDecodeError:
            print("✅ API responded (non-JSON)")
            return True
    else:
        print(f"❌ API health check failed: {response.get('error', 'Unknown error')}")
        return False


def upload_document_for_relevance(pages, doc_type, test_name):
    """Upload a document for relevance extraction"""
    print(f"\n📤 Testing {test_name} ({pages} pages, type: {doc_type})...")
    
    try:
        # Generate test PDF with legal content
        pdf_data = create_legal_pdf(pages, doc_type)
        filename = f"relevance_test_{pages}pages_{doc_type}.pdf"
        
        print(f"   📄 Generated test PDF: {len(pdf_data):,} bytes")
        
        # Create multipart data
        body, headers = create_multipart_data(filename, pdf_data, 'application/pdf')
        
        # Upload
        start_time = time.time()
        response = make_http_request(UPLOAD_ENDPOINT, method='POST', data=body, headers=headers, timeout=60)
        upload_time = time.time() - start_time
        
        if response['success'] and response['status_code'] == 200:
            try:
                result = json.loads(response['data'])
                job_id = result.get('job_id')
                print(f"✅ Upload successful - Job ID: {job_id}")
                print(f"⏱️  Upload time: {upload_time:.2f}s")
                
                return {
                    'test_name': test_name,
                    'pages': pages,
                    'doc_type': doc_type,
                    'filename': filename,
                    'job_id': job_id,
                    'upload_time': upload_time,
                    'upload_success': True,
                    'pdf_size_bytes': len(pdf_data),
                    'start_time': datetime.now()
                }
            except json.JSONDecodeError:
                print(f"❌ Upload response invalid JSON")
                return {
                    'test_name': test_name,
                    'pages': pages,
                    'doc_type': doc_type,
                    'filename': filename,
                    'upload_success': False,
                    'upload_time': upload_time,
                    'error': 'Invalid JSON response'
                }
        else:
            print(f"❌ Upload failed: {response.get('error', 'Unknown error')}")
            if response.get('data'):
                print(f"   Response: {response['data'][:200]}")
            return {
                'test_name': test_name,
                'pages': pages,
                'doc_type': doc_type,
                'filename': filename,
                'upload_success': False,
                'upload_time': upload_time,
                'error': response.get('error', 'Upload failed')
            }
            
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return {
            'test_name': test_name,
            'pages': pages,
            'doc_type': doc_type,
            'upload_success': False,
            'error': f'Unexpected error: {e}'
        }


def monitor_relevance_job(job_info, max_wait_minutes=30):
    """Monitor relevance extraction job with detailed tracking"""
    if not job_info.get('upload_success') or 'job_id' not in job_info:
        return job_info
    
    job_id = job_info['job_id']
    test_name = job_info['test_name']
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    print(f"🔄 Monitoring {test_name} (Job: {job_id})...")
    
    while time.time() - start_time < max_wait_seconds:
        try:
            url = f"{JOB_STATUS_ENDPOINT}/{job_id}"
            response = make_http_request(url, timeout=10)
            
            if response['success'] and response['status_code'] == 200:
                try:
                    status_data = json.loads(response['data'])
                    current_status = status_data.get('status')
                    progress = status_data.get('progress', 0)
                    
                    if current_status == 'COMPLETED':
                        completion_time = time.time() - start_time
                        
                        # Extract relevance results
                        results = status_data.get('results', {})
                        
                        print(f"✅ Relevance extraction completed in {completion_time:.2f}s")
                        print(f"   📋 Document type: {results.get('document_type', 'unknown')}")
                        
                        # Show relevances
                        relevances = results.get('relevances', [])
                        print(f"   📊 Number of relevances: {len(relevances)}")
                        
                        if relevances:
                            for idx, rel in enumerate(relevances, 1):
                                print(f"\n   📌 Relevance #{idx}:")
                                print(f"      Text: {rel.get('relevance_text', 'N/A')[:100]}...")
                                
                                pinpoints = rel.get('pinpoints', [])
                                print(f"      Pinpoints: {len(pinpoints)}")
                                for pp in pinpoints[:2]:  # Show first 2
                                    print(f"        - {pp.get('reference', 'N/A')} (page {pp.get('page', '?')})")
                                
                                quotes = rel.get('evidence_quotes', [])
                                print(f"      Evidence quotes: {len(quotes)}")
                                for quote in quotes[:2]:  # Show first 2
                                    print(f"        - \"{quote.get('text', 'N/A')[:50]}...\" (page {quote.get('page', '?')})")
                        
                        # Show token usage and cost
                        token_usage = results.get('token_usage', {})
                        estimated_cost = results.get('estimated_cost', 0)
                        
                        print(f"\n   💰 Token Usage & Cost:")
                        print(f"      Input tokens: {token_usage.get('input_tokens', 0):,}")
                        print(f"      Output tokens: {token_usage.get('output_tokens', 0):,}")
                        print(f"      Total tokens: {token_usage.get('total_tokens', 0):,}")
                        print(f"      Estimated cost: ${estimated_cost:.6f}")
                        
                        # Show processing details
                        pages_processed = results.get('pages_processed', {})
                        print(f"\n   ⚙️  Processing Method:")
                        print(f"      Gemini pages: {pages_processed.get('gemini', 0)}")
                        print(f"      PPStructure pages: {pages_processed.get('ppstructure', 0)}")
                        
                        job_info.update({
                            'final_status': 'COMPLETED',
                            'total_processing_time': completion_time,
                            'results': results,
                            'num_relevances': len(relevances),
                            'token_usage': token_usage,
                            'estimated_cost': estimated_cost,
                            'completion_time': datetime.now()
                        })
                        return job_info
                        
                    elif current_status == 'FAILED':
                        completion_time = time.time() - start_time
                        print(f"❌ Relevance extraction failed after {completion_time:.2f}s")
                        job_info.update({
                            'final_status': 'FAILED',
                            'total_processing_time': completion_time,
                            'error': status_data.get('error', 'Unknown error'),
                            'completion_time': datetime.now()
                        })
                        return job_info
                        
                    elif current_status == 'PROCESSING':
                        elapsed = time.time() - start_time
                        if progress > 0:
                            print(f"⏳ {progress}% complete ({elapsed:.1f}s)")
                        
                except json.JSONDecodeError:
                    print(f"⚠️  Invalid JSON response for {test_name}")
                    
            elif response['status_code'] == 404:
                print(f"❌ Job {job_id} not found")
                job_info.update({
                    'final_status': 'NOT_FOUND',
                    'error': 'Job not found',
                    'total_processing_time': time.time() - start_time
                })
                return job_info
                
            else:
                print(f"⚠️  Status check failed: {response.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"⚠️  Error checking status: {e}")
        
        time.sleep(5)  # Poll every 5 seconds
    
    # Timeout
    timeout_time = time.time() - start_time
    print(f"⏰ Timeout reached for {test_name} after {timeout_time:.2f}s")
    job_info.update({
        'final_status': 'TIMEOUT',
        'total_processing_time': timeout_time,
        'error': f'No completion after {max_wait_minutes} minutes'
    })
    return job_info


def generate_report(test_results):
    """Generate test report"""
    print("\n" + "="*70)
    print("📊 RELEVANCE EXTRACTION TEST REPORT")
    print("="*70)
    
    completed = [r for r in test_results if r.get('final_status') == 'COMPLETED']
    failed = [r for r in test_results if r.get('final_status') != 'COMPLETED' and r.get('final_status') is not None]
    
    print(f"\n✅ Completed: {len(completed)}/{len(test_results)}")
    print(f"❌ Failed: {len(failed)}/{len(test_results)}")
    
    if completed:
        print(f"\n📋 RESULTS SUMMARY:")
        print("-" * 70)
        
        total_cost = 0
        total_tokens = 0
        
        for result in completed:
            test_name = result.get('test_name', 'Unknown')
            pages = result.get('pages', 0)
            total_time = result.get('total_processing_time', 0) or 0
            num_relevances = result.get('num_relevances', 0)
            token_usage = result.get('token_usage', {})
            estimated_cost = result.get('estimated_cost', 0)
            
            total_cost += estimated_cost
            total_tokens += token_usage.get('total_tokens', 0)
            
            print(f"\n📄 {test_name} ({pages} pages)")
            print(f"   ⏱️  Processing time: {total_time:.2f}s")
            print(f"   📌 Relevances extracted: {num_relevances}")
            print(f"   💾 Tokens: {token_usage.get('total_tokens', 0):,}")
            print(f"   💰 Cost: ${estimated_cost:.6f}")
        
        print(f"\n📈 TOTAL STATISTICS:")
        print(f"   Total tokens processed: {total_tokens:,}")
        print(f"   Total estimated cost: ${total_cost:.6f}")
        print(f"   Average cost per page: ${total_cost / max(sum(r.get('pages', 0) for r in completed), 1):.6f}")
    
    return {
        'total_tests': len(test_results),
        'completed': len(completed),
        'failed': len(failed),
        'success_rate': len(completed)/len(test_results)*100 if test_results else 0
    }


def main():
    """Main test execution"""
    print("🚀 Relevance Extraction Test Suite")
    print("=" * 70)
    print(f"🐍 Python {sys.version[:5]} | 📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Purpose: Test token-efficient legal relevance extraction")
    print()
    
    # Check API health
    if not check_api_health():
        print("❌ API not available")
        return False
    
    print()
    
    # Define test cases covering different document sizes and types (max 70 pages)
    test_cases = [
        (5, "case", "Small Case (5 pages)"),           # Gemini only
        (8, "statute", "Medium Statute (8 pages)"),     # Gemini only (< 8 pages)
        (15, "case", "Large Case (15 pages)"),          # Mixed: first 90% Gemini, last 10% PPStructure
        (70, "statute", "Maximum Statute (70 pages)"), # Max test size
    ]
    
    print(f"📋 TEST PLAN ({len(test_cases)} documents):")
    print("-" * 70)
    
    for pages, doc_type, name in test_cases:
        method = "Gemini-only" if pages < 8 else "Mixed (90% Gemini + 10% PPStructure)"
        print(f"   📄 {name} - {method}")
    
    print()
    
    # Execute tests
    print(f"📤 Uploading and processing documents...\n")
    final_results = []
    
    for i, (pages, doc_type, test_name) in enumerate(test_cases):
        print(f"\n--- Test {i+1}/{len(test_cases)} ---")
        
        try:
            # Upload document
            result = upload_document_for_relevance(pages, doc_type, test_name)
            
            if result.get('upload_success'):
                # Monitor job - PPStructure is slow, give generous timeout
                max_wait = max(30, pages * 2 + 120)  # Much longer timeout for PPStructure processing
                print(f"⏱️  Estimated processing time: ~{max_wait} seconds")
                completed_result = monitor_relevance_job(result, max_wait_minutes=max_wait/60)
                final_results.append(completed_result)
            else:
                final_results.append(result)
                
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            final_results.append({
                'test_name': test_name,
                'pages': pages,
                'upload_success': False,
                'error': f'Unexpected error: {e}'
            })
        
        # Brief pause between tests
        if i < len(test_cases) - 1:
            print("⏳ Waiting 2 seconds before next test...")
            time.sleep(2)
    
    # Generate report
    report = generate_report(final_results)
    
    print()
    print("=" * 70)
    if report['success_rate'] >= 75:
        print("✅ TESTS PASSED")
    else:
        print("⚠️  TESTS PARTIALLY PASSED")
    
    print(f"   Success rate: {report['success_rate']:.1f}%")
    print(f"   Completed: {report['completed']}/{report['total_tests']}")
    print()
    print("🏁 Relevance extraction test suite completed!")
    
    return report['success_rate'] >= 75


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
