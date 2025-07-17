#!/usr/bin/env python3
"""
Test script for the new single document upload API endpoint.
This demonstrates how other systems should integrate with your API.
"""

import requests
import time
import os
import json

def test_single_upload():
    """Test the new /api/upload endpoint"""
    
    # API base URL (adjust as needed)
    base_url = "http://localhost:8000"
    
    # Test with a sample file (create a proper dummy PDF if needed)
    test_file_path = "test_document.pdf"
    
    # Create a proper minimal PDF file for testing if it doesn't exist
    if not os.path.exists(test_file_path):
        # Create a minimal but valid PDF file
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 5 0 R
>>
>>
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj

5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj

xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000281 00000 n 
0000000373 00000 n 
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
459
%%EOF"""
        
        with open(test_file_path, 'wb') as f:
            f.write(pdf_content)
        print(f"Created proper test PDF: {test_file_path}")
    
    try:
        print("🚀 Testing Single Document Upload API")
        print("=" * 50)
        
        # 1. Upload the document
        print("📄 Uploading document...")
        with open(test_file_path, 'rb') as f:
            files = {'file': (test_file_path, f, 'application/pdf')}
            response = requests.post(f"{base_url}/api/upload", files=files)
        
        if response.status_code != 200:
            print(f"❌ Upload failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        upload_result = response.json()
        print("✅ Upload successful!")
        print(f"   Job ID: {upload_result['job_id']}")
        print(f"   Filename: {upload_result['filename']}")
        print(f"   Polling endpoint: {upload_result['polling_endpoint']}")
        print(f"   Estimated completion: {upload_result['estimated_completion_minutes']} minutes")
        
        # 2. Poll for completion
        job_id = upload_result['job_id']
        print(f"\n🔄 Polling for completion...")
        
        max_attempts = 120  # 30 minutes max
        attempt = 0
        
        while attempt < max_attempts:
            response = requests.get(f"{base_url}/api/job/{job_id}")
            
            if response.status_code != 200:
                print(f"❌ Status check failed: {response.status_code}")
                break
            
            status_data = response.json()
            
            # Print debug info about the API response
            print(f"   Status: {status_data.get('status', 'unknown')} ({status_data.get('progress', 0)}%)")
            if 'stage' in status_data:
                print(f"   Current stage: {status_data['stage']}")
            elif 'message' in status_data:
                print(f"   Current stage: {status_data['message']}")
            
            if status_data.get('status') == 'COMPLETED':
                print("✅ Processing completed!")
                
                # DEBUG: Print the full response to see what's available
                print("\n🔍 DEBUG: Full API Response:")
                print(json.dumps(status_data, indent=2))
                
                # Extract results from the response
                print(f"\n📋 Results:")
                
                # Get basic info from API response
                total_pages = status_data.get('total_pages', 'N/A')
                processing_time = status_data.get('processing_time_seconds', 'N/A')
                
                # NEW: Extract results from the enhanced API response
                results_obj = status_data.get('results', {})
                
                if results_obj:
                    # Results are now included directly in the API response!
                    summary = results_obj.get('summary', 'N/A')
                    date = results_obj.get('date', 'N/A')
                    total_pages = results_obj.get('total_pages', total_pages)  # Use results value if available
                    processing_completed_at = results_obj.get('processing_completed_at', 'N/A')
                    
                    print(f"   ✅ Results included directly in API response (production-ready!)")
                    
                    # Key extracted information that other systems actually need
                    extracted_info = results_obj.get('extracted_info', {})
                    
                else:
                    # Fallback: Try to read detailed results from the results file (legacy approach)
                    summary = 'N/A'
                    date = 'N/A'
                    processing_completed_at = 'N/A'
                    
                    results_path = status_data.get('results_path')
                    
                    if results_path and os.path.exists(results_path):
                        try:
                            with open(results_path, 'r', encoding='utf-8') as f:
                                detailed_results = json.load(f)
                            summary = detailed_results.get('summary', 'N/A')
                            date = detailed_results.get('date', 'N/A')
                            processing_completed_at = detailed_results.get('processing_completed_at', 'N/A')
                            print(f"   ⚠️ Using legacy file-based results from: {results_path}")
                        except Exception as e:
                            print(f"   ❌ Could not read results file: {e}")
                
                print(f"   📅 Date: {date}")
                print(f"   📄 Summary: {summary}")
                print(f"   📊 Total pages: {total_pages}")
                print(f"   ✅ Completed at: {processing_completed_at}")
                
                # Show key extracted information if available
                if results_obj and extracted_info:
                    print(f"   🔍 Extracted info: {', '.join(extracted_info.keys())}")
                    
                    # Show key extracted information
                    if 'key_dates' in extracted_info:
                        print(f"     - Key dates: {extracted_info['key_dates']}")
                    if 'main_parties' in extracted_info:
                        print(f"     - Main parties: {extracted_info['main_parties']}")
                
                break
            
            elif status_data.get('status') == 'FAILED':
                print(f"❌ Processing failed: {status_data.get('error', 'Unknown error')}")
                break
            
            # Wait before next check
            time.sleep(15)
            attempt += 1
        
        if attempt >= max_attempts:
            print("⏰ Timeout waiting for completion")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            print(f"\n🧹 Cleaned up test file: {test_file_path}")
    
    print("\n" + "=" * 50)
    print("Testing with real PDF files (if available):")
    
    # Check for any PDF files in the current directory
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF files: {pdf_files}")
        print("To test with a real PDF, modify the test_file_path variable above.")
    else:
        print("No PDF files found in current directory.")
        print("To test with a real PDF, place a PDF file in this directory and run again.")

if __name__ == "__main__":
    test_single_upload() 