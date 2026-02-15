#!/usr/bin/env python3
"""
PDF Length Processing Test - Built-in Libraries Only
Law Document Processing System

This script tests the /api/upload endpoint with dynamically generated PDFs
of different lengths to analyze chunking strategies and response times.

Expected behavior based on API documentation:
- 1-10 pages: Gemini-only processing (fast, sequential)
- 11+ pages: Hybrid processing (35% PPStructure + 65% Gemini sequential)

Usage: python3 simple_api_test_builtin_fixed.py
"""

#TODO: parellel polling for results

import urllib.request
import urllib.parse
import urllib.error
import json
import time
import os
import sys
import psutil
import traceback
from datetime import datetime

# API Configuration
BASE_URL = "http://204.12.246.205:8000"
UPLOAD_ENDPOINT = f"{BASE_URL}/api/upload"
JOB_STATUS_ENDPOINT = f"{BASE_URL}/api/job"
HEALTH_ENDPOINT = f"{BASE_URL}/health"

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
UPLOAD_ENDPOINT = f"{BASE_URL}/api/upload"
JOB_STATUS_ENDPOINT = f"{BASE_URL}/api/job"
HEALTH_ENDPOINT = f"{BASE_URL}/health"

print(f"ℹ️  Using API Base URL: {BASE_URL}")

# Load API key from environment
API_KEY = os.getenv('API_KEYS', '').split(',')[0] if os.getenv('API_KEYS') else None
if API_KEY:
    print(f"✅ API key loaded: {API_KEY[:10]}...")
else:
    print("⚠️ No API key found in environment - authentication may fail")

# at the top of your file, add these imports:
import io
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    USE_REPORTLAB = True
    print("✅ PDF generation via ReportLab enabled")
except ImportError:
    USE_REPORTLAB = False
    print("⚠️ reportlab not installed; falling back to minimal generator")

# Excel generation imports
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    USE_EXCEL = True
    print("✅ Excel generation via openpyxl enabled")
except ImportError:
    USE_EXCEL = False
    print("⚠️ openpyxl not installed; Excel report disabled")


def log_memory_usage(context=""):
    """Log current memory usage"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        print(f"🧠 Memory usage {context}: {memory_mb:.1f} MB")
        
        # Check if memory usage is getting high (over 2GB)
        if memory_mb > 2048:
            print(f"⚠️  HIGH MEMORY USAGE: {memory_mb:.1f} MB")
        
        return memory_mb
    except Exception as e:
        print(f"⚠️  Could not check memory usage: {e}")
        return 0

def create_minimal_pdf(num_pages, content_per_page="Sample legal document content"):
    """Create a sample legal PDF of `num_pages`, using ReportLab if available."""
    log_memory_usage(f"before creating {num_pages}-page PDF")
    
    try:
        if USE_REPORTLAB:
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=letter)
            for i in range(num_pages):
                # Log memory every 100 pages for large documents
                if num_pages > 100 and i % 100 == 0:
                    log_memory_usage(f"after page {i} of {num_pages}")
                
                text = (
                    f"Page {i+1}\n"
                    f"{content_per_page}\n"
                    f"This is page {i+1} of {num_pages}."
                )
                for line_no, line in enumerate(text.split("\n")):
                    c.drawString(72, 720 - line_no * 14, line)
                c.showPage()
            c.save()
            pdf_bytes = buf.getvalue()
            buf.close()
            log_memory_usage(f"after creating {num_pages}-page PDF")
            print(f"   📄 PDF created via ReportLab: {len(pdf_bytes):,} bytes")
            return pdf_bytes

        # fallback to your original minimal generator
        print("   ⚠️ Using fallback minimal PDF generator")
        pdf_objects = []
        pdf_content = "%PDF-1.4\n"
        # Catalog
        pdf_objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        # Pages
        refs = " ".join(f"{3+i} 0 R" for i in range(num_pages))
        pdf_objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{refs}] /Count {num_pages} >>\nendobj\n")
        # Page objs
        for i in range(num_pages):
            pdf_objects.append(
                f"{3+i} 0 obj\n"
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {3+num_pages+i} 0 R >>\nendobj\n"
            )
        # Content streams
        for i in range(num_pages):
            stream = f"Page {i+1}\\n{content_per_page}\\nThis is page {i+1} of {num_pages}."
            content_stream = f"BT /F1 12 Tf 72 720 Td ({stream}) Tj ET"
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
        log_memory_usage(f"after creating {num_pages}-page PDF (fallback)")
        return pdf_content.encode("utf-8")
    except MemoryError as e:
        print(f"❌ MEMORY ERROR creating {num_pages}-page PDF: {e}")
        raise
    except Exception as e:
        print(f"❌ ERROR creating {num_pages}-page PDF: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        raise


def create_multipart_data(filename, file_data, content_type):
    """Create multipart/form-data for file upload using only built-in libraries"""
    boundary = f"----WebKitFormBoundary{int(time.time() * 1000)}"
    
    # Build multipart body
    body_parts = []
    
    # File part
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

def upload_pdf_test(pages, test_name):
    """Upload a generated PDF and measure response time"""
    print(f"\n📤 Testing {test_name} ({pages} pages)...")
    log_memory_usage("before PDF generation")
    
    try:
        # Generate PDF
        pdf_data = create_minimal_pdf(pages, f"Test content for {test_name}")
        filename = f"test_{pages}pages.pdf"
        
        print(f"   📄 Generated PDF: {len(pdf_data)} bytes")
        log_memory_usage("after PDF generation")
        
        # Create multipart data
        body, headers = create_multipart_data(filename, pdf_data, 'application/pdf')
        log_memory_usage("after multipart data creation")
        
        # Upload with timing
        upload_start = time.time()
        response = make_http_request(UPLOAD_ENDPOINT, method='POST', data=body, headers=headers, timeout=60)
        upload_end = time.time()
        
        upload_time = upload_end - upload_start
        log_memory_usage("after upload")
        
        if response['success'] and response['status_code'] == 200:
            try:
                result = json.loads(response['data'])
                job_id = result.get('job_id')
                print(f"✅ Upload successful - Job ID: {job_id}")
                print(f"⏱️  Upload time: {upload_time:.2f}s")
                
                return {
                    'test_name': test_name,
                    'pages': pages,
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
                'filename': filename,
                'upload_success': False,
                'upload_time': upload_time,
                'error': response.get('error', 'Upload failed')
            }
    except MemoryError as e:
        print(f"❌ MEMORY ERROR during upload test for {pages} pages: {e}")
        return {
            'test_name': test_name,
            'pages': pages,
            'upload_success': False,
            'error': f'Memory error: {e}'
        }
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR during upload test for {pages} pages: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return {
            'test_name': test_name,
            'pages': pages,
            'upload_success': False,
            'error': f'Unexpected error: {e}'
        }

def monitor_job_with_timing(job_info, max_wait_minutes=30):
    """Monitor job with detailed timing and processing method detection"""
    if not job_info.get('upload_success') or 'job_id' not in job_info:
        return job_info
    
    job_id = job_info['job_id']
    test_name = job_info['test_name']
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    print(f"🔄 Monitoring {test_name} (Job: {job_id})...")
    
    # Track processing stages and times
    stage_times = {}
    first_processing_time = None
    completion_time = None
    
    while time.time() - start_time < max_wait_seconds:
        try:
            url = f"{JOB_STATUS_ENDPOINT}/{job_id}"
            status_start = time.time()
            response = make_http_request(url, timeout=10)
            status_time = time.time() - status_start
            
            if response['success'] and response['status_code'] == 200:
                try:
                    status_data = json.loads(response['data'])
                    current_status = status_data.get('status')
                    progress = status_data.get('progress', 0)
                    stage = status_data.get('stage', 'Unknown')
                    
                    # Track first processing time
                    if current_status == 'PROCESSING' and first_processing_time is None:
                        first_processing_time = time.time() - start_time
                        print(f"⚡ Processing started after {first_processing_time:.2f}s")
                    
                    if current_status == 'COMPLETED':
                        completion_time = time.time() - start_time
                        
                        # Extract detailed results
                        results = status_data.get('results', {})
                        total_pages = results.get('total_pages', 'unknown')
                        processing_method = status_data.get('processing_method', 'unknown')
                        ppstructure_chunks = status_data.get('ppstructure_chunks', 0)
                        gemini_chunks = status_data.get('gemini_chunks', 0)
                        
                        print(f"✅ Completed in {completion_time:.2f}s")
                        print(f"   Method: {processing_method}")
                        print(f"   Chunks: {ppstructure_chunks} PPStructure + {gemini_chunks} Gemini")
                        
                        job_info.update({
                            'final_status': 'COMPLETED',
                            'total_processing_time': completion_time,
                            'first_processing_time': first_processing_time,
                            'stage_times': stage_times,
                            'results': results,
                            'processing_method': processing_method,
                            'ppstructure_chunks': ppstructure_chunks,
                            'gemini_chunks': gemini_chunks,
                            'confirmed_pages': total_pages,
                            'api_response_time_avg': status_time,
                            'completion_time': datetime.now()
                        })
                        return job_info
                        
                    elif current_status == 'FAILED':
                        completion_time = time.time() - start_time
                        print(f"❌ Failed after {completion_time:.2f}s")
                        job_info.update({
                            'final_status': 'FAILED',
                            'total_processing_time': completion_time,
                            'first_processing_time': first_processing_time,
                            'error': status_data.get('error', 'Unknown error'),
                            'completion_time': datetime.now()
                        })
                        return job_info
                        
                    elif current_status == 'PROCESSING':
                        elapsed = time.time() - start_time
                        if progress > 0:
                            print(f"⏳ {progress}% complete ({elapsed:.1f}s)")
                        
                    else:
                        elapsed = time.time() - start_time
                        print(f"📋 Status: {current_status} ({elapsed:.1f}s)")
                        
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
        
        time.sleep(8)  # Poll every 8 seconds for more responsive monitoring
    
    # Timeout
    timeout_time = time.time() - start_time
    print(f"⏰ Timeout reached for {test_name} after {timeout_time:.2f}s")
    job_info.update({
        'final_status': 'TIMEOUT',
        'total_processing_time': timeout_time,
        'first_processing_time': first_processing_time,
        'error': f'No completion after {max_wait_minutes} minutes'
    })
    return job_info

def generate_simple_report(test_results):
    """Generate a simple processing report"""
    print("\n" + "="*60)
    print("📊 PROCESSING RESULTS")
    print("="*60)
    
    completed = [r for r in test_results if r.get('final_status') == 'COMPLETED']
    failed = [r for r in test_results if r.get('final_status') != 'COMPLETED']
    
    print(f"✅ Completed: {len(completed)}/{len(test_results)}")
    print(f"❌ Failed: {len(failed)}/{len(test_results)}")
    
    if completed:
        print(f"\n📋 DOCUMENT PROCESSING SUMMARY:")
        print("-" * 60)
        
        for result in completed:
            pages = result.get('pages', 0)
            total_time = result.get('total_processing_time', 0) or 0
            method = result.get('processing_method', 'unknown')
            pp_chunks = result.get('ppstructure_chunks', 0)
            gem_chunks = result.get('gemini_chunks', 0)
            
            rate = pages / max(total_time, 0.1)
            
            print(f"📄 {pages} pages: {total_time:.1f}s ({rate:.1f} pages/sec)")
            print(f"   Method: {method}")
            print(f"   Chunks: {pp_chunks} PP + {gem_chunks} Gemini")
            print()
    
    return {
        'total_tests': len(test_results),
        'completed': len(completed),
        'success_rate': len(completed)/len(test_results)*100 if test_results else 0
    }

def generate_excel_report(test_results, timestamp):
    """Generate Excel report with timing data and error tracking"""
    if not USE_EXCEL:
        print("⚠️ Excel generation not available (openpyxl not installed)")
        return None
    
    filename = f"processing_report_{timestamp}.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Processing Results"
    
    # Headers
    headers = [
        "Document", "Pages", "Upload Time (s)", "Total Processing Time (s)", 
        "Processing Method", "PPStructure Chunks", "Gemini Chunks", 
        "Status", "Error", "Pages/Second", "Start Time", "End Time"
    ]
    
    # Style headers
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
    
    # Data rows
    for row, result in enumerate(test_results, 2):
        ws.cell(row=row, column=1, value=result.get('test_name', 'Unknown'))
        ws.cell(row=row, column=2, value=result.get('pages', 0))
        
        upload_time = result.get('upload_time', 0)
        ws.cell(row=row, column=3, value=round(upload_time, 2) if upload_time else 0)
        
        total_time = result.get('total_processing_time', 0)
        ws.cell(row=row, column=4, value=round(total_time, 2) if total_time else 0)
        
        ws.cell(row=row, column=5, value=result.get('processing_method', 'Unknown'))
        ws.cell(row=row, column=6, value=result.get('ppstructure_chunks', 0))
        ws.cell(row=row, column=7, value=result.get('gemini_chunks', 0))
        
        status = result.get('final_status', 'FAILED' if not result.get('upload_success') else 'UNKNOWN')
        ws.cell(row=row, column=8, value=status)
        
        error = result.get('error', '')
        ws.cell(row=row, column=9, value=error if error else 'None')
        
        # Calculate pages per second
        pages = result.get('pages', 0)
        if total_time and total_time > 0 and pages > 0:
            pages_per_sec = pages / total_time
            ws.cell(row=row, column=10, value=round(pages_per_sec, 2))
        else:
            ws.cell(row=row, column=10, value=0)
        
        # Start and end times
        start_time = result.get('start_time')
        if start_time:
            ws.cell(row=row, column=11, value=start_time.strftime('%Y-%m-%d %H:%M:%S'))
        
        end_time = result.get('completion_time')
        if end_time:
            ws.cell(row=row, column=12, value=end_time.strftime('%Y-%m-%d %H:%M:%S'))
        
        # Color code status
        status_cell = ws.cell(row=row, column=8)
        if status == 'COMPLETED':
            status_cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
        elif status in ['FAILED', 'TIMEOUT', 'NOT_FOUND']:
            status_cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    
    # Auto-adjust column widths
    for column in ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        ws.column_dimensions[column_letter].width = adjusted_width
    
    # Add summary sheet
    summary_ws = wb.create_sheet("Summary")
    
    completed = [r for r in test_results if r.get('final_status') == 'COMPLETED']
    failed = [r for r in test_results if r.get('final_status') != 'COMPLETED']
    
    summary_data = [
        ["Metric", "Value"],
        ["Total Documents", len(test_results)],
        ["Completed", len(completed)],
        ["Failed", len(failed)],
        ["Success Rate", f"{(len(completed)/len(test_results)*100):.1f}%" if test_results else "0%"],
        ["", ""],
        ["Average Processing Time (Completed)", f"{sum(r.get('total_processing_time', 0) for r in completed) / len(completed):.2f}s" if completed else "N/A"],
        ["Average Pages/Second (Completed)", f"{sum(r.get('pages', 0) / max(r.get('total_processing_time', 0.1), 0.1) for r in completed) / len(completed):.2f}" if completed else "N/A"]
    ]
    
    for row, (metric, value) in enumerate(summary_data, 1):
        summary_ws.cell(row=row, column=1, value=metric).font = Font(bold=True)
        summary_ws.cell(row=row, column=2, value=value)
    
    summary_ws.column_dimensions['A'].width = 30
    summary_ws.column_dimensions['B'].width = 20
    
    try:
        wb.save(filename)
        print(f"📊 Excel report saved: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Failed to save Excel report: {e}")
        return None

def main():
    """Main test execution - PDF length processing analysis"""
    print("🚀 PDF Length Processing Analysis - Stress Test Up to 1000 Pages")
    print("=" * 70)
    print(f"🐍 Python {sys.version[:5]} | 📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📚 Using libraries: urllib, json, time, os, psutil (for memory monitoring)")
    print("📄 Testing PDF processing with different page counts - STRESS TEST MODE")
    
    # Log initial memory usage
    log_memory_usage("at startup")
    
    # Check API health
    if not check_api_health():
        print("❌ API is not available. Exiting.")
        return
    
    # Define test cases - capped at 70 pages max
    test_cases = [
        (5, "5-Page Document"),       # Should use Gemini only
        (10, "10-Page Document"),     # Should use Gemini only (boundary)
        (50, "50-Page Document"),     # Should use Hybrid processing
        (70, "70-Page Document"),     # Maximum test size
    ]
    
    print(f"\n📋 TEST PLAN ({len(test_cases)} documents):")
    print("-" * 50)
    
    for pages, name in test_cases:
        method = "Gemini only" if pages <= 10 else "Hybrid processing"
        print(f"   📄 {pages} pages - {method}")
    
    estimated_minutes = len(test_cases) * 5 + sum(max(2, pages // 20) for pages, _ in test_cases)
    print(f"\n⏱️  Estimated time: ~{estimated_minutes} minutes (stress test with memory monitoring)")
    print("-" * 50)
    
    # Process documents sequentially (upload and wait for completion)
    print(f"\n📤 Processing documents sequentially...")
    final_results = []
    
    for i, (pages, test_name) in enumerate(test_cases):
        print(f"\n--- Processing Document {i+1}/{len(test_cases)} ---")
        print(f"📄 {test_name} ({pages} pages)")
        
        try:
            # Upload document
            result = upload_pdf_test(pages, test_name)
            
            if result.get('upload_success'):
                # Immediately monitor this job to completion
                # PPStructure is slow, so give generous timeouts
                max_wait = max(30, pages // 2 + 120)  # Much longer timeout for PPStructure processing
                print(f"⏱️  Estimated processing time: ~{max_wait} minutes")
                completed_result = monitor_job_with_timing(result, max_wait_minutes=max_wait)
                final_results.append(completed_result)
            else:
                final_results.append(result)
        except MemoryError as e:
            print(f"❌ MEMORY ERROR processing {pages}-page document: {e}")
            final_results.append({
                'test_name': test_name,
                'pages': pages,
                'upload_success': False,
                'error': f'Memory error: {e}'
            })
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR processing {pages}-page document: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            final_results.append({
                'test_name': test_name,
                'pages': pages,
                'upload_success': False,
                'error': f'Unexpected error: {e}'
            })
        
        # Short delay between documents
        if i < len(test_cases) - 1:
            print("⏳ Waiting 3 seconds before next document...")
            time.sleep(3)
    
    # Generate simple report
    report_stats = generate_simple_report(final_results)
    
    # Save detailed results
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_filename = f"pdf_length_analysis_{timestamp}.json"
        
        # Prepare JSON-serializable results
        json_results = []
        for result in final_results:
            json_result = {}
            for key, value in result.items():
                if isinstance(value, datetime):
                    json_result[key] = value.isoformat()
                else:
                    json_result[key] = value
            json_results.append(json_result)
        
        # Add summary stats
        output_data = {
            'test_metadata': {
                'timestamp': timestamp,
                'total_tests': len(test_cases),
                'api_base_url': BASE_URL,
                'python_version': sys.version
            },
            'summary_stats': report_stats,
            'test_results': json_results
        }
        
        with open(results_filename, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"\n💾 Detailed results saved to: {results_filename}")
        
        # Generate Excel report
        excel_filename = generate_excel_report(final_results, timestamp)
        if excel_filename:
            print(f"📊 Excel report saved to: {excel_filename}")
        
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    # Final summary
    success_rate = report_stats.get('success_rate', 0)
    success_icon = "✅" if success_rate >= 80 else "⚠️"
    
    print(f"\n{success_icon} SUMMARY:")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Completed: {report_stats.get('completed', 0)}/{report_stats.get('total_tests', 0)}")
    print(f"\n🏁 Test completed!")

if __name__ == "__main__":
    main() 