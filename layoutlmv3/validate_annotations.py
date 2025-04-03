import os
import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns

def validate_annotations(annotation_dir):
    """
    Validate and analyze the quality of generated annotations.
    """
    results = {
        'total_documents': 0,
        'total_annotations': 0,
        'matched_annotations': 0,
        'confidence_distribution': defaultdict(int),
        'words_per_annotation': [],
        'annotations_per_document': [],
        'match_types': defaultdict(int),
        'question_types': defaultdict(lambda: {'total': 0, 'matched': 0})
    }
    
    files = [f for f in os.listdir(annotation_dir) if f.endswith('_layoutlm.json')]
    results['total_documents'] = len(files)
    
    print(f"Analyzing {len(files)} annotation files...")
    
    for file in tqdm(files, desc="Analyzing files"):
        try:
            with open(os.path.join(annotation_dir, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            annotations = data.get('annotations', [])
            results['annotations_per_document'].append(len(annotations))
            results['total_annotations'] += len(annotations)
            
            for annotation in annotations:
                words = annotation.get('words', [])
                if words:
                    results['matched_annotations'] += 1
                    results['words_per_annotation'].append(len(words))
                    
                    # Extract question type from ID
                    question_id = annotation.get('id', '')
                    question_type = None
                    if '_' in question_id:
                        parts = question_id.split('_')
                        if len(parts) >= 2:
                            question_type = parts[-2]
                    
                    if question_type:
                        results['question_types'][question_type]['total'] += 1
                        results['question_types'][question_type]['matched'] += 1
                    
                    # Analyze confidence
                    confidences = []
                    for word in words:
                        conf = word.get('confidence', 0)
                        confidences.append(conf)
                        conf_bucket = round(conf * 10) / 10  # Round to nearest 0.1
                        results['confidence_distribution'][conf_bucket] += 1
                    
                    # Count match types
                    match_type = annotation.get('match_confidence', 'standard')
                    results['match_types'][match_type] += 1
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    # Print summary
    print("\n=== Annotation Validation Summary ===")
    print(f"Documents analyzed: {results['total_documents']}")
    print(f"Total annotations: {results['total_annotations']}")
    if results['total_annotations'] > 0:
        print(f"Successfully matched: {results['matched_annotations']} ({results['matched_annotations']/results['total_annotations']*100:.1f}%)")
    else:
        print("No annotations found.")
        return results
    
    print("\nMatch types:")
    for match_type, count in results['match_types'].items():
        print(f"  {match_type}: {count} ({count/results['matched_annotations']*100:.1f}%)")
    
    # Sort question types by match rate
    question_type_stats = []
    for qtype, stats in results['question_types'].items():
        match_rate = stats['matched'] / stats['total'] if stats['total'] > 0 else 0
        question_type_stats.append((qtype, stats['total'], stats['matched'], match_rate))
    
    question_type_stats.sort(key=lambda x: x[3], reverse=True)
    
    if question_type_stats:
        print("\nTop 10 Question Types by Match Rate:")
        for qtype, total, matched, rate in question_type_stats[:10]:
            print(f"  {qtype}: {matched}/{total} ({rate*100:.1f}%)")
        
        if len(question_type_stats) > 10:
            print("\nBottom 10 Question Types by Match Rate:")
            for qtype, total, matched, rate in question_type_stats[-10:]:
                print(f"  {qtype}: {matched}/{total} ({rate*100:.1f}%)")
    
    # Generate visualizations
    try:
        os.makedirs(os.path.join(annotation_dir, 'analysis'), exist_ok=True)
        
        # Plot confidence distribution
        plt.figure(figsize=(12, 8))
        conf_values = sorted(results['confidence_distribution'].keys())
        conf_counts = [results['confidence_distribution'][k] for k in conf_values]
        plt.bar([str(x) for x in conf_values], conf_counts)
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('OCR Confidence Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(annotation_dir, 'analysis', 'confidence_distribution.png'))
        
        # Plot annotations per document
        plt.figure(figsize=(12, 8))
        plt.hist(results['annotations_per_document'], bins=20)
        plt.xlabel('Annotations per Document')
        plt.ylabel('Number of Documents')
        plt.title('Annotation Distribution Across Documents')
        plt.savefig(os.path.join(annotation_dir, 'analysis', 'annotation_distribution.png'))
        
        # Plot words per annotation
        plt.figure(figsize=(12, 8))
        plt.hist(results['words_per_annotation'], bins=50)
        plt.xlabel('Words per Annotation')
        plt.ylabel('Number of Annotations')
        plt.title('Words per Annotation Distribution')
        plt.savefig(os.path.join(annotation_dir, 'analysis', 'words_per_annotation.png'))
        
        # Plot question type match rates
        if question_type_stats:
            plt.figure(figsize=(15, 10))
            question_types = [x[0] for x in question_type_stats]
            match_rates = [x[3] * 100 for x in question_type_stats]
            
            # Only show top 20 for readability
            if len(question_types) > 20:
                question_types = question_types[:20]
                match_rates = match_rates[:20]
                
            # Create horizontal bar chart
            y_pos = np.arange(len(question_types))
            plt.barh(y_pos, match_rates)
            plt.yticks(y_pos, question_types)
            plt.xlabel('Match Rate (%)')
            plt.title('Match Rate by Question Type (Top 20)')
            plt.tight_layout()
            plt.savefig(os.path.join(annotation_dir, 'analysis', 'question_type_match_rates.png'))
        
        # Save detailed results to JSON
        with open(os.path.join(annotation_dir, 'analysis', 'validation_results.json'), 'w', encoding='utf-8') as f:
            # Convert defaultdicts to regular dicts for serialization
            serializable_results = {
                'total_documents': results['total_documents'],
                'total_annotations': results['total_annotations'],
                'matched_annotations': results['matched_annotations'],
                'confidence_distribution': {str(k): v for k, v in results['confidence_distribution'].items()},
                'words_per_annotation_stats': {
                    'min': min(results['words_per_annotation']) if results['words_per_annotation'] else 0,
                    'max': max(results['words_per_annotation']) if results['words_per_annotation'] else 0,
                    'mean': sum(results['words_per_annotation'])/len(results['words_per_annotation']) if results['words_per_annotation'] else 0,
                },
                'annotations_per_document_stats': {
                    'min': min(results['annotations_per_document']) if results['annotations_per_document'] else 0,
                    'max': max(results['annotations_per_document']) if results['annotations_per_document'] else 0,
                    'mean': sum(results['annotations_per_document'])/len(results['annotations_per_document']) if results['annotations_per_document'] else 0,
                },
                'match_types': dict(results['match_types']),
                'question_types': {
                    qtype: {
                        'total': stats['total'],
                        'matched': stats['matched'],
                        'match_rate': stats['matched'] / stats['total'] if stats['total'] > 0 else 0
                    }
                    for qtype, stats in results['question_types'].items()
                }
            }
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nAnalysis results and visualizations saved to {os.path.join(annotation_dir, 'analysis')}")
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate LayoutLMv3 annotations')
    parser.add_argument('--dir', type=str, default='CUAD_v1/layoutlmv3',
                        help='Directory containing annotation JSON files')
    args = parser.parse_args()
    
    validate_annotations(args.dir) 