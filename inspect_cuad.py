import json
import sys

# Load the dataset
with open('CUAD_v1/CUAD_v1.json', 'r', encoding='utf-8') as f:
    cuad_data = json.load(f)

# Print dataset structure
print("Top level keys:", list(cuad_data.keys()))

# Inspect data structure
if 'data' in cuad_data:
    print("Number of documents:", len(cuad_data['data']))
    print("\nFirst document keys:", list(cuad_data['data'][0].keys()))
    
    # Look for paragraphs
    doc = cuad_data['data'][0]
    if 'paragraphs' in doc:
        print("\nParagraphs keys:", list(doc['paragraphs'][0].keys()))
        
        # Look for QAs
        paragraph = doc['paragraphs'][0]
        if 'qas' in paragraph:
            print("\nQA structure:", list(paragraph['qas'][0].keys()))
            
            # Look inside answers
            qa = paragraph['qas'][0]
            if 'answers' in qa and qa['answers']:
                print("\nAnswer structure:", list(qa['answers'][0].keys()))
                print("\nSample answer text:", qa['answers'][0]['text'][:100] + "..." if len(qa['answers'][0]['text']) > 100 else qa['answers'][0]['text'])
    else:
        print("\nNo 'paragraphs' key in document")
else:
    print("No 'data' key in the dataset") 