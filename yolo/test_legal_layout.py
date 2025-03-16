import os
import cv2
import argparse
import logging
from yolo.document_layout import DocumentLayoutAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def test_document_layout(image_path, model_path=None, output_dir='layout_results'):
    """Test document layout detection on a single image"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename without extension for output files
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Initialize the document layout analyzer
    logger.info(f"Initializing document layout analyzer{'with custom model' if model_path else ''}")
    analyzer = DocumentLayoutAnalyzer(model_path=model_path, confidence_threshold=0.25)
    
    # Load the image
    logger.info(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return
    
    # Detect document regions
    logger.info("Detecting document regions")
    regions = analyzer.detect_regions(img)
    
    # Extract region images
    logger.info(f"Extracting {len(regions)} region images")
    region_images = analyzer.get_region_images(img, regions)
    
    # Create visualization
    logger.info("Creating visualization")
    vis_img = analyzer.visualize_regions(img, regions)
    
    # Save visualization
    vis_path = os.path.join(output_dir, f"{base_filename}_regions.jpg")
    cv2.imwrite(vis_path, vis_img)
    logger.info(f"Visualization saved to {vis_path}")
    
    # Save individual region images
    for i, region in enumerate(regions):
        region_img = region_images[i]
        region_path = os.path.join(output_dir, f"{base_filename}_region_{i}_{region.region_type}.jpg")
        cv2.imwrite(region_path, region_img)
    
    # Generate document map
    logger.info("Generating document map")
    doc_map = analyzer.generate_document_map(regions)
    
    # Save document map
    import json
    map_path = os.path.join(output_dir, f"{base_filename}_map.json")
    with open(map_path, 'w') as f:
        json.dump(doc_map, f, indent=2)
    logger.info(f"Document map saved to {map_path}")
    
    # Return results
    return {
        'regions': regions,
        'visualization_path': vis_path,
        'map_path': map_path
    }

def main():
    """Main function for testing document layout detection"""
    parser = argparse.ArgumentParser(description='Test document layout detection on legal documents')
    parser.add_argument('image_path', help='Path to the document image or PDF')
    parser.add_argument('--model', help='Path to a custom document layout model')
    parser.add_argument('--output', default='layout_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if the legal model exists
    legal_model_path = os.path.join('models', 'legal_layout_model.pt')
    if os.path.exists(legal_model_path) and args.model is None:
        logger.info(f"Using legal document model: {legal_model_path}")
        args.model = legal_model_path
    
    # Test document layout detection
    results = test_document_layout(
        image_path=args.image_path,
        model_path=args.model,
        output_dir=args.output
    )
    
    if results:
        logger.info(f"Detected {len(results['regions'])} regions")
        logger.info(f"Results saved to {args.output}")
        
        # Print region types and counts
        region_types = {}
        for region in results['regions']:
            region_types[region.region_type] = region_types.get(region.region_type, 0) + 1
        
        logger.info("Region type counts:")
        for rtype, count in region_types.items():
            logger.info(f"  - {rtype}: {count}")

if __name__ == "__main__":
    main() 