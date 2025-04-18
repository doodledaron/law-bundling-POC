#!/usr/bin/env python3
import re
import sys
import os
import json
from collections import defaultdict

def update_document_data(all_documents, partial_matches, doc, success_rate, qa_successful, qa_total):
    """Helper function to consistently update document data across all collections."""
    # Update all_documents first
    if doc in all_documents:
        all_documents[doc]['reported_success_rate'] = success_rate
        all_documents[doc]['qa_pairs_successful'] = qa_successful
        all_documents[doc]['qa_pairs_total'] = qa_total
        
        # If this document is in partial_matches and has 100% success rate, remove it
        if doc in partial_matches and success_rate == 100:
            print(f"Removing {doc} from missing annotations list because it has 100% success rate")
            del partial_matches[doc]
        # Otherwise update the partial_matches data to stay consistent
        elif doc in partial_matches:
            partial_matches[doc]['reported_success_rate'] = success_rate
            partial_matches[doc]['qa_pairs_successful'] = qa_successful
            partial_matches[doc]['qa_pairs_total'] = qa_total

def analyze_log_file(log_path):
    """Analyze a processing log file and extract information about failed annotations."""
    print(f"Analyzing log file: {log_path}")
    
    # Dictionary to store failed documents and their annotations
    failed_documents = {}
    
    # Dictionary to store documents with partial annotation matches
    partial_matches = {}
    
    # Dictionary to store success rates for all documents
    all_documents = {}
    
    # Track current document being processed
    current_doc = None
    
    # Debugging: track if we found any success rates
    success_rate_count = 0
    
    # Regular expressions to extract information
    doc_pattern = re.compile(r'Processing document \d+/\d+: (.+)')
    doc_file_pattern = re.compile(r'Processing document: (.+\.pdf)')
    success_match_pattern = re.compile(r'Successfully matched (\d+)/(\d+) annotations')
    success_rate_pattern = re.compile(r'Success rate: ([\d.]+)% \((\d+)/(\d+) QA pairs\)')
    # More generic timestamp pattern
    timestamp_success_pattern = re.compile(r'.* Success rate: ([\d.]+)% \((\d+)/(\d+) QA pairs\)')
    # Even more basic pattern matching just the numbers
    basic_success_pattern = re.compile(r'Success rate: (\d+\.\d+)%.*\((\d+)/(\d+)')
    failed_annotation_pattern = re.compile(r'Could not find match for answer: (.{1,50})\.{3}?')
    warning_pattern = re.compile(r'Low confidence match for: (.{1,50})\.{3}?')
    
    # Open and read the log file
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    # Print total lines for debugging
    print(f"Total lines in log file: {len(lines)}")
    
    # Process each line in the log
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Extract document name
        doc_match = doc_pattern.search(line)
        if doc_match:
            current_doc = doc_match.group(1)
            continue
        
        # Extract document file path
        doc_file_match = doc_file_pattern.search(line)
        if doc_file_match and current_doc:
            doc_file = doc_file_match.group(1)
            if current_doc not in failed_documents:
                failed_documents[current_doc] = {
                    'file_path': doc_file,
                    'failed_annotations': [],
                    'warnings': []
                }
            if current_doc not in all_documents:
                all_documents[current_doc] = {
                    'file_path': doc_file,
                    'successful': 0,
                    'total': 0,
                    'reported_success_rate': 0,
                    'qa_pairs_successful': 0,
                    'qa_pairs_total': 0
                }
            continue
        
        # Special check for success rate lines - add debug print for any line containing "Success rate"
        if "Success rate" in line and current_doc:
            print(f"Found success rate line: {line}")
        
        # Extract annotations matched information
        match_success = success_match_pattern.search(line)
        if match_success and current_doc:
            successful = int(match_success.group(1))
            total = int(match_success.group(2))
            
            # Store annotation match info
            if current_doc in all_documents:
                all_documents[current_doc]['successful'] = successful
                all_documents[current_doc]['total'] = total
            
            # Only add to partial_matches if we don't already know it has 100% success rate
            current_success_rate = all_documents.get(current_doc, {}).get('reported_success_rate', 0)
            if successful < total and current_success_rate != 100:
                # If already in partial_matches, update it
                if current_doc in partial_matches:
                    partial_matches[current_doc].update({
                        'successful': successful,
                        'total': total,
                        'missing': total - successful,
                    })
                else:
                    # Add new entry
                    partial_matches[current_doc] = {
                        'successful': successful,
                        'total': total,
                        'missing': total - successful,
                        'reported_success_rate': current_success_rate,
                        'qa_pairs_successful': all_documents.get(current_doc, {}).get('qa_pairs_successful', 0),
                        'qa_pairs_total': all_documents.get(current_doc, {}).get('qa_pairs_total', 0),
                    }
            continue
        
        # Extract reported success rate information (from the log)
        success_rate_match = success_rate_pattern.search(line)
        if success_rate_match and current_doc:
            success_rate = float(success_rate_match.group(1))
            qa_successful = int(success_rate_match.group(2))
            qa_total = int(success_rate_match.group(3))
            
            success_rate_count += 1
            print(f"Matched standard success rate for {current_doc}: {success_rate}% ({qa_successful}/{qa_total})")
            
            # Use helper function to consistently update data
            update_document_data(all_documents, partial_matches, current_doc, success_rate, qa_successful, qa_total)
            continue
        
        # Try the timestamped success rate pattern
        timestamp_success_match = timestamp_success_pattern.search(line)
        if timestamp_success_match and current_doc:
            success_rate = float(timestamp_success_match.group(1))
            qa_successful = int(timestamp_success_match.group(2))
            qa_total = int(timestamp_success_match.group(3))
            
            success_rate_count += 1
            print(f"Matched timestamped success rate for {current_doc}: {success_rate}% ({qa_successful}/{qa_total})")
            
            # Use helper function to consistently update data
            update_document_data(all_documents, partial_matches, current_doc, success_rate, qa_successful, qa_total)
            continue
        
        # Try the most basic success rate pattern as fallback
        basic_match = basic_success_pattern.search(line)
        if basic_match and current_doc:
            success_rate = float(basic_match.group(1))
            qa_successful = int(basic_match.group(2))
            qa_total = int(basic_match.group(3))
            
            success_rate_count += 1
            print(f"Matched basic success rate for {current_doc}: {success_rate}% ({qa_successful}/{qa_total})")
            
            # Use helper function to consistently update data
            update_document_data(all_documents, partial_matches, current_doc, success_rate, qa_successful, qa_total)
            continue
        
        # Extract failed annotation
        failed_match = failed_annotation_pattern.search(line)
        if failed_match and current_doc:
            failed_text = failed_match.group(1)
            failed_documents[current_doc]['failed_annotations'].append(failed_text)
            continue
        
        # Extract low confidence warnings
        warning_match = warning_pattern.search(line)
        if warning_match and current_doc:
            warning_text = warning_match.group(1)
            failed_documents[current_doc]['warnings'].append(warning_text)
            continue
    
    # Remove documents with no failures or warnings
    for doc in list(failed_documents.keys()):
        if not failed_documents[doc]['failed_annotations'] and not failed_documents[doc]['warnings']:
            del failed_documents[doc]
    
    # Remove documents with 100% success rate from partial_matches
    for doc in list(partial_matches.keys()):
        if doc in all_documents and all_documents[doc].get('reported_success_rate', 0) == 100:
            print(f"Removing {doc} from missing annotations list because it has 100% success rate")
            del partial_matches[doc]
    
    # Update partial_matches with accurate data from all_documents
    for doc in list(partial_matches.keys()):
        if doc in all_documents:
            # Copy success rate and QA pair data from all_documents
            partial_matches[doc]['reported_success_rate'] = all_documents[doc].get('reported_success_rate', 0)
            partial_matches[doc]['qa_pairs_successful'] = all_documents[doc].get('qa_pairs_successful', 0)
            partial_matches[doc]['qa_pairs_total'] = all_documents[doc].get('qa_pairs_total', 0)
            
            # Debug print
            print(f"Updated missing annotations data for {doc}: {partial_matches[doc]['reported_success_rate']}% "
                  f"({partial_matches[doc]['qa_pairs_successful']}/{partial_matches[doc]['qa_pairs_total']})")
    
    # Print debug info
    print(f"Found {success_rate_count} success rate entries in the log")
    print(f"Processed {len(all_documents)} documents")
    
    # Check for documents missing success rate info
    missing_success_rate = [doc for doc, info in all_documents.items() 
                          if info.get('reported_success_rate', 0) == 0 and info.get('total', 0) > 0]
    if missing_success_rate:
        print(f"WARNING: {len(missing_success_rate)} documents are missing success rate information:")
        for doc in missing_success_rate[:5]:  # Show at most 5
            print(f"  - {doc}")
        if len(missing_success_rate) > 5:
            print(f"  - ... and {len(missing_success_rate) - 5} more")
    
    # Final pass: scan for success rates in a continuous block of lines
    print("\nPerforming final pass to find missed success rates...")
    doc_blocks = {}
    current_block = []
    current_block_doc = None
    
    for line in lines:
        line = line.strip()
        
        # Start of a new document block
        doc_match = doc_pattern.search(line)
        if doc_match:
            # Save previous block if any
            if current_block_doc and current_block:
                doc_blocks[current_block_doc] = current_block
            
            # Start new block
            current_block_doc = doc_match.group(1)
            current_block = [line]
        elif current_block_doc:
            # Continue adding to current block
            current_block.append(line)
    
    # Add the last block
    if current_block_doc and current_block:
        doc_blocks[current_block_doc] = current_block
    
    # Scan each document's log block for success rate information
    for doc, block in doc_blocks.items():
        if doc in all_documents and all_documents[doc]['reported_success_rate'] == 0:
            # This document doesn't have success rate yet, search the block
            for line in block:
                # Try different patterns for success rate
                if "Success rate:" in line:
                    print(f"Found success rate line for {doc}: {line}")
                    # Extract numbers manually
                    try:
                        # Extract percentage
                        rate_part = line.split("Success rate:")[1].strip()
                        success_rate = float(rate_part.split("%")[0].strip())
                        
                        # Extract QA pairs
                        qa_part = rate_part.split("(")[1].split(")")[0]
                        qa_successful, qa_total = map(int, qa_part.split("/"))
                        
                        print(f"Manually parsed: {success_rate}% ({qa_successful}/{qa_total})")
                        
                        # Use helper function to consistently update data
                        update_document_data(all_documents, partial_matches, doc, success_rate, qa_successful, qa_total)
                        
                        break  # Found what we needed
                    except Exception as e:
                        print(f"Error parsing success rate: {e}")
    
    # Check again for documents missing success rate
    missing_success_rate = [doc for doc, info in all_documents.items() 
                          if info.get('reported_success_rate', 0) == 0 and info.get('total', 0) > 0]
    if missing_success_rate:
        print(f"AFTER FINAL PASS: {len(missing_success_rate)} documents still missing success rate information")
    else:
        print("SUCCESS: All documents now have success rate information")
    
    return failed_documents, partial_matches, all_documents

def generate_report(failed_documents, partial_matches, all_documents):
    """Generate a report from the analysis results."""
    # Summary statistics
    total_docs = len(set(list(failed_documents.keys()) + list(partial_matches.keys())))
    total_failed_annotations = sum(len(doc['failed_annotations']) for doc in failed_documents.values())
    total_warnings = sum(len(doc['warnings']) for doc in failed_documents.values())
    total_missing = sum(match.get('missing', 0) for match in partial_matches.values())
    
    # Get overall counts from log
    total_annotations = sum(doc.get('total', 0) for doc in all_documents.values())
    successful_annotations = sum(doc.get('successful', 0) for doc in all_documents.values())
    
    # Get QA pair stats
    total_qa_pairs = sum(doc.get('qa_pairs_total', 0) for doc in all_documents.values())
    successful_qa_pairs = sum(doc.get('qa_pairs_successful', 0) for doc in all_documents.values())
    
    # Calculate average success rate from reported rates (if available)
    doc_with_rates = [doc for doc in all_documents.values() if doc.get('reported_success_rate', 0) > 0]
    avg_success_rate = sum(doc.get('reported_success_rate', 0) for doc in doc_with_rates) / len(doc_with_rates) if doc_with_rates else 0
    
    # Create report
    report = {
        "summary": {
            "total_documents": len(all_documents),
            "total_documents_with_issues": total_docs,
            "total_failed_annotations": total_failed_annotations,
            "total_low_confidence_warnings": total_warnings,
            "total_missing_annotations": total_missing,
            "total_annotations": total_annotations,
            "successful_annotations": successful_annotations,
            "total_qa_pairs": total_qa_pairs,
            "successful_qa_pairs": successful_qa_pairs,
            "average_success_rate": avg_success_rate
        },
        "documents_with_missing_annotations": [
            {
                "document": doc,
                "successful": info.get("successful", 0),
                "total": info.get("total", 0),
                "missing": info.get("missing", 0),
                "reported_success_rate": info.get("reported_success_rate", 0),
                "qa_pairs_successful": info.get("qa_pairs_successful", 0),
                "qa_pairs_total": info.get("qa_pairs_total", 0),
                "note": info.get("note", "")
            }
            for doc, info in partial_matches.items()
        ],
        "documents_with_failures": [
            {
                "document": doc,
                "file_path": info["file_path"],
                "failed_annotations": info["failed_annotations"],
                "low_confidence_warnings": info["warnings"]
            }
            for doc, info in failed_documents.items()
        ],
        "all_documents": [
            {
                "document": doc,
                "file_path": info["file_path"],
                "successful": info.get("successful", 0),
                "total": info.get("total", 0),
                "reported_success_rate": info.get("reported_success_rate", 0),
                "qa_pairs_successful": info.get("qa_pairs_successful", 0),
                "qa_pairs_total": info.get("qa_pairs_total", 0)
            }
            for doc, info in all_documents.items()
        ]
    }
    
    return report

def write_output(report, output_path):
    """Write the report to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {output_path}")
    
    # Also create a simplified CSV-like text file for quick reference
    txt_path = output_path.replace('.json', '.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=== DOCUMENTS WITH MISSING ANNOTATIONS ===\n")
        f.write("Document Name,Annotations(Success/Total),Missing,QA Pairs(Success/Total),Success Rate(%)\n")
        for doc in report["documents_with_missing_annotations"]:
            qa_str = f"{doc.get('qa_pairs_successful', 0)}/{doc.get('qa_pairs_total', 0)}"
            f.write(f"{doc['document']},{doc['successful']}/{doc['total']},{doc['missing']},{qa_str},{doc.get('reported_success_rate', 0):.2f}%\n")
        
        f.write("\n=== DOCUMENTS WITH FAILED ANNOTATIONS ===\n")
        f.write("Document Name,File Path,Failed Annotations,Warnings\n")
        for doc in report["documents_with_failures"]:
            f.write(f"{doc['document']},{doc['file_path']},{len(doc['failed_annotations'])},{len(doc['low_confidence_warnings'])}\n")
        
        f.write("\n=== ALL DOCUMENTS SUMMARY ===\n")
        f.write("Document Name,Annotations(Success/Total),QA Pairs(Success/Total),Success Rate(%)\n")
        for doc in sorted(report["all_documents"], key=lambda x: x.get("reported_success_rate", 0)):
            ann_str = f"{doc.get('successful', 0)}/{doc.get('total', 0)}"
            qa_str = f"{doc.get('qa_pairs_successful', 0)}/{doc.get('qa_pairs_total', 0)}"
            
            # Only include documents with annotations or QA pairs
            if doc.get('total', 0) > 0 or doc.get('qa_pairs_total', 0) > 0:
                f.write(f"{doc['document']},{ann_str},{qa_str},{doc.get('reported_success_rate', 0):.2f}%\n")
        
    print(f"Simplified report written to {txt_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_processing_log.py <log_file_path> [output_file_path]")
        sys.exit(1)
    
    log_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # Default output path based on input filename
        base_name = os.path.splitext(os.path.basename(log_path))[0]
        output_path = f"{base_name}_analysis.json"
    
    # Analyze the log file
    failed_documents, partial_matches, all_documents = analyze_log_file(log_path)
    
    # Generate and write the report
    report = generate_report(failed_documents, partial_matches, all_documents)
    write_output(report, output_path)
    
    # Print summary to console
    print("\nANALYSIS SUMMARY:")
    print(f"Total documents analyzed: {report['summary']['total_documents']}")
    print(f"Documents with issues: {report['summary']['total_documents_with_issues']}")
    print(f"Total annotations: {report['summary']['successful_annotations']}/{report['summary']['total_annotations']} (successful/total)")
    print(f"Total QA pairs: {report['summary']['successful_qa_pairs']}/{report['summary']['total_qa_pairs']} (successful/total)")
    print(f"Average reported success rate: {report['summary']['average_success_rate']:.2f}%")
    print(f"Total failed annotations: {report['summary']['total_failed_annotations']}")
    print(f"Total low confidence warnings: {report['summary']['total_low_confidence_warnings']}")
    print(f"Total missing annotations: {report['summary']['total_missing_annotations']}")
    
    # Print top 5 documents with lowest success rates
    print("\nTOP 5 DOCUMENTS WITH LOWEST SUCCESS RATES:")
    docs_with_rate = [doc for doc in report["all_documents"] if doc.get("reported_success_rate", 0) > 0]
    sorted_by_rate = sorted(docs_with_rate, key=lambda x: x.get("reported_success_rate", 0))
    for i, doc in enumerate(sorted_by_rate[:5]):
        qa_str = f"{doc.get('qa_pairs_successful', 0)}/{doc.get('qa_pairs_total', 0)}"
        print(f"{i+1}. {doc['document']}: {qa_str} QA pairs, {doc.get('reported_success_rate', 0):.2f}% success rate")
    
    # Print top 5 documents with most missing annotations
    if report["documents_with_missing_annotations"]:
        print("\nTOP 5 DOCUMENTS WITH MOST MISSING ANNOTATIONS:")
        sorted_docs = sorted(report["documents_with_missing_annotations"], 
                           key=lambda x: x.get("missing", 0), reverse=True)
        for i, doc in enumerate(sorted_docs[:5]):
            qa_str = f"{doc.get('qa_pairs_successful', 0)}/{doc.get('qa_pairs_total', 0)}"
            print(f"{i+1}. {doc['document']}: {doc.get('missing', 0)} missing, {qa_str} QA pairs, {doc.get('reported_success_rate', 0):.2f}% success rate")

if __name__ == "__main__":
    main() 