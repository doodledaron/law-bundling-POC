#!/usr/bin/env python3
"""
Container Warmup Test Script
Law Document Processing System

Minimal test script to warm up all containers after docker compose up.
Based on simple_api_test.py but simplified for quick container verification.

Usage: python3 warmup_test.py
"""

import urllib.request
import urllib.parse
import urllib.error
import json
import time
import sys
from datetime import datetime

# API Configuration
BASE_URL = "http://204.12.246.205:8000"
UPLOAD_ENDPOINT = f"{BASE_URL}/api/upload"
JOB_STATUS_ENDPOINT = f"{BASE_URL}/api/job"
HEALTH_ENDPOINT = f"{BASE_URL}/health"

def make_http_request(url, method='GET', data=None, headers=None, timeout=30):
    """Make HTTP request using urllib"""
    try:
        if headers is None:
            headers = {}
        
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

def create_minimal_pdf(pages=2):
    """Create a minimal PDF for testing"""
    pdf_content = "%PDF-1.4\n"
    
    # Simple PDF structure for warmup test
    pdf_objects = []
    
    # Catalog
    pdf_objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    
    # Pages object
    refs = " ".join(f"{3+i} 0 R" for i in range(pages))
    pdf_objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{refs}] /Count {pages} >>\nendobj\n")
    
    # Page objects
    for i in range(pages):
        pdf_objects.append(
            f"{3+i} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {3+pages+i} 0 R >>\nendobj\n"
        )
    
    # Content streams
    for i in range(pages):
        stream = f"Warmup test page {i+1}"
        content_stream = f"BT /F1 12 Tf 72 720 Td ({stream}) Tj ET"
        pdf_objects.append(
            f"{3+pages+i} 0 obj\n<< /Length {len(content_stream)} >>\nstream\n"
            f"{content_stream}\nendstream\nendobj\n"
        )
    
    pdf_content += "".join(pdf_objects)
    
    # Cross-reference table
    xref_offset = len(pdf_content)
    pdf_content += "xref\n"
    pdf_content += f"0 {len(pdf_objects)+1}\n0000000000 65535 f \n"
    
    offset = len("%PDF-1.4\n")
    for obj in pdf_objects:
        pdf_content += f"{offset:010d} 00000 n \n"
        offset += len(obj)
    
    # Trailer
    pdf_content += (
        "trailer\n"
        f"<< /Size {len(pdf_objects)+1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    
    return pdf_content.encode("utf-8")

def create_multipart_data(filename, file_data, content_type):
    """Create multipart/form-data for file upload"""
    boundary = f"----WebKitFormBoundary{int(time.time() * 1000)}"
    
    # Build multipart body
    body_parts = []
    body_parts.append(f'--{boundary}')
    body_parts.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"')
    body_parts.append(f'Content-Type: {content_type}')
    body_parts.append('')
    
    # Convert body parts to bytes
    body_header = '\r\n'.join(body_parts).encode('utf-8')
    body_footer = f'\r\n--{boundary}--\r\n'.encode('utf-8')
    
    # Combine all parts
    body = body_header + b'\r\n' + file_data + body_footer
    
    headers = {
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'Content-Length': str(len(body))
    }
    
    return body, headers

def warmup_upload_test(test_num=1, pages=2):
    """Simple upload test to warm up containers"""
    print(f"📤 Testing upload #{test_num} to warm containers ({pages} pages)...")
    
    try:
        # Create small test PDF
        pdf_data = create_minimal_pdf(pages)
        filename = f"warmup_test_{test_num}.pdf"
        
        print(f"   📄 Generated test PDF: {len(pdf_data)} bytes")
        
        # Create multipart data
        body, headers = create_multipart_data(filename, pdf_data, 'application/pdf')
        
        # Upload
        start_time = time.time()
        response = make_http_request(UPLOAD_ENDPOINT, method='POST', data=body, headers=headers, timeout=30)
        upload_time = time.time() - start_time
        
        if response['success'] and response['status_code'] == 200:
            try:
                result = json.loads(response['data'])
                job_id = result.get('job_id')
                print(f"✅ Upload #{test_num} successful - Job ID: {job_id}")
                print(f"⏱️  Upload time: {upload_time:.2f}s")
                return job_id
            except json.JSONDecodeError:
                print(f"❌ Upload #{test_num} response invalid JSON")
                return None
        else:
            print(f"❌ Upload #{test_num} failed: {response.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Upload test #{test_num} failed: {e}")
        return None

def check_job_status(job_id, test_num=1, max_wait_seconds=60):
    """Check job status briefly to warm processing containers"""
    if not job_id:
        return False
        
    print(f"🔄 Checking job #{test_num} status for warmup...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait_seconds:
        try:
            url = f"{JOB_STATUS_ENDPOINT}/{job_id}"
            response = make_http_request(url, timeout=10)
            
            if response['success'] and response['status_code'] == 200:
                try:
                    status_data = json.loads(response['data'])
                    current_status = status_data.get('status')
                    
                    if current_status == 'COMPLETED':
                        elapsed = time.time() - start_time
                        print(f"✅ Job #{test_num} completed in {elapsed:.1f}s")
                        return True
                    elif current_status == 'FAILED':
                        elapsed = time.time() - start_time
                        print(f"❌ Job #{test_num} failed after {elapsed:.1f}s")
                        return False
                    elif current_status == 'PROCESSING':
                        elapsed = time.time() - start_time
                        print(f"⏳ Job #{test_num} processing... ({elapsed:.1f}s)")
                    else:
                        elapsed = time.time() - start_time
                        print(f"📋 Job #{test_num} status: {current_status} ({elapsed:.1f}s)")
                        
                except json.JSONDecodeError:
                    print(f"⚠️  Job #{test_num} invalid JSON response")
                    
            elif response['status_code'] == 404:
                print(f"❌ Job #{test_num} not found")
                return False
            else:
                print(f"⚠️  Job #{test_num} status check failed: {response.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"⚠️  Job #{test_num} error checking status: {e}")
        
        time.sleep(3)  # Check every 3 seconds for faster warmup
    
    print(f"⏰ Job #{test_num} warmup timeout reached")
    return False

def main():
    """Main warmup test execution"""
    print("🚀 Container Warmup Test - All 6 Workers")
    print("=" * 50)
    print(f"🐍 Python {sys.version[:5]} | 📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Purpose: Warm up all 6 law-worker-documents containers")
    print("📦 Containers: law-worker-documents-container-1 through 6")
    print()
    
    # Check API health
    if not check_api_health():
        print("❌ API not available. Containers may still be starting.")
        print("💡 Try running this script again in a few minutes.")
        return False
    
    print()
    print("🔥 Starting warmup sequence for all 6 worker containers...")
    print("📋 Uploading 6 documents to ensure each worker gets activated")
    print()
    
    # Upload multiple jobs to warm all 6 worker containers
    jobs = []
    total_successes = 0
    
    for i in range(1, 7):  # 1 through 6
        print(f"--- Worker Container #{i} Warmup ---")
        
        # Vary document sizes to trigger different processing paths
        pages = 2 if i <= 3 else 12  # Small docs for first 3, larger for last 3
        job_id = warmup_upload_test(test_num=i, pages=pages)
        
        if job_id:
            jobs.append((job_id, i))
            total_successes += 1
        
        # Small delay between uploads to spread across workers
        if i < 6:
            print("⏳ Waiting 2 seconds before next upload...")
            time.sleep(2)
        
        print()
    
    print(f"📊 Upload Summary: {total_successes}/6 jobs submitted successfully")
    print()
    
    if jobs:
        print("🔄 Monitoring jobs to warm processing containers...")
        print("⚡ This will activate all worker containers")
        print()
        
        completed_jobs = 0
        failed_jobs = 0
        
        # Monitor all jobs with shorter timeout since we have multiple
        for job_id, test_num in jobs:
            success = check_job_status(job_id, test_num=test_num, max_wait_seconds=120)
            if success:
                completed_jobs += 1
            else:
                failed_jobs += 1
            
            # Brief pause between job checks
            time.sleep(1)
        
        print()
        print("=" * 50)
        print("📊 WARMUP RESULTS")
        print("=" * 50)
        print(f"✅ Jobs completed: {completed_jobs}/6")
        print(f"❌ Jobs failed/timeout: {failed_jobs}/6")
        print(f"📤 Upload success rate: {total_successes}/6")
        
        if completed_jobs >= 4:  # At least 4 out of 6 containers warmed
            print()
            print("🎉 Warmup SUCCESS!")
            print("✅ Majority of worker containers are warmed up and ready")
            print("🚀 System ready for production workloads")
        elif total_successes >= 4:
            print()
            print("⚠️  Warmup PARTIAL SUCCESS")
            print("✅ Upload containers warmed, some processing containers may need more time")
            print("💡 Most containers should be ready for lighter workloads")
        else:
            print()
            print("❌ Warmup INCOMPLETE")
            print("⚠️  Many containers may not be ready")
            print("💡 Check container status: docker compose ps")
    else:
        print("❌ No jobs were submitted successfully")
        print("💡 Check if all containers are running with: docker compose ps")
    
    print()
    print("🏁 Multi-container warmup test finished!")
    return total_successes >= 4

if __name__ == "__main__":
    main()