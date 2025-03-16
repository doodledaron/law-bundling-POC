import os
import logging
import yaml
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Ensure model directory exists
os.makedirs('models', exist_ok=True)

# Define output path for model
OUTPUT_MODEL_PATH = 'models/legal_layout_model.pt'

def train_model():
    """Train a YOLOv8 model for legal document layout detection"""
    
    logger.info("Setting up training for legal document layout model")
    
    # Download YOLOv8 model to start from
    logger.info("Downloading base YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Load a smaller model for faster training
    
    # Create dataset configuration file
    dataset_yaml = """
# Legal Document Layout Detection Dataset Configuration
path: ./legal_document_dataset  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')

# Classes
names:
  0: paragraph
  1: heading
  2: table
  3: list
  4: figure
  5: caption
  6: header
  7: footer
  8: signature
  9: stamp
  10: footnote
  11: page_number
  12: title
  13: watermark
    """
    
    # Save the dataset configuration
    with open('legal_document_dataset.yaml', 'w') as f:
        f.write(dataset_yaml)
    
    logger.info("Dataset configuration written to legal_document_dataset.yaml")
    
    # Check if we have a dataset
    if not os.path.exists('./legal_document_dataset'):
        logger.warning("Dataset directory not found. Creating sample structure.")
        os.makedirs('./legal_document_dataset/train/images', exist_ok=True)
        os.makedirs('./legal_document_dataset/train/labels', exist_ok=True)
        os.makedirs('./legal_document_dataset/val/images', exist_ok=True)
        os.makedirs('./legal_document_dataset/val/labels', exist_ok=True)
        
        logger.info("""
Please add your legal document training data to the legal_document_dataset directory.
The directory structure should be:
- legal_document_dataset/
  - train/
    - images/  # JPG or PNG files
    - labels/  # YOLO format txt files (class x_center y_center width height)
  - val/
    - images/
    - labels/
        """)
        
        logger.info("For an existing pre-trained legal document model, consider using DocBank, PubLayNet, or FUNSD datasets.")
        logger.info("Alternatively, download pre-trained models from Hugging Face.")
        return False
    
    # Train the model
    logger.info("Starting model training...")
    try:
        # Train the model for layout detection
        results = model.train(
            data='legal_document_dataset.yaml',
            epochs=100,
            imgsz=640,
            batch=16,
            patience=20,
            name='legal_layout',
            device='0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
        )
        
        # Save the model
        model_path = os.path.join('runs', 'detect', 'legal_layout', 'weights', 'best.pt')
        if os.path.exists(model_path):
            import shutil
            shutil.copy(model_path, OUTPUT_MODEL_PATH)
            logger.info(f"Model saved to {OUTPUT_MODEL_PATH}")
            return True
        else:
            logger.error(f"Training completed but model file not found at {model_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False

def download_pretrained_model():
    """Download a pre-trained document layout model"""
    import requests
    import shutil
    
    logger.info("Downloading pre-trained document layout model")
    
    # There's no direct YOLOv8 legal document layout model publicly available,
    # but here are some options to consider:
    
    # 1. Microsoft's Layout-Parser models (not YOLOv8 but specialized for documents)
    # 2. DocBank models
    # 3. PubLayNet models adapted to YOLOv8
    
    # For demo, let's use the general PubLayNet Faster-RCNN model (for layout detection)
    # and convert it to YOLOv8 format
    
    logger.info("This function would download a pre-trained legal document model.")
    logger.info("Recommended options:")
    logger.info("1. DocBank: https://github.com/doc-analysis/DocBank")
    logger.info("2. PubLayNet: https://github.com/ibm-aur-nlp/PubLayNet")
    logger.info("3. DocLayNet: https://github.com/microsoft/unilm/tree/master/doclaynet")
    logger.info("4. LegalLayoutDetection: https://huggingface.co/datasets")
    
    # For an actual implementation, you would:
    # 1. Download the model from the appropriate source
    # 2. Convert it to YOLOv8 format if needed
    # 3. Save it to models/legal_layout_model.pt
    
    logger.info("To use with your code:")
    logger.info("1. Download a pre-trained document layout model")
    logger.info("2. Convert it to YOLOv8 format if necessary")
    logger.info("3. Place it at: models/legal_layout_model.pt")
    
    return False

def main():
    """Main function to set up legal document layout model"""
    
    logger.info("Legal Document Layout Model Setup")
    
    # Check if the model already exists
    if os.path.exists(OUTPUT_MODEL_PATH):
        logger.info(f"Model already exists at {OUTPUT_MODEL_PATH}")
        return
    
    # Ask the user what they want to do
    print("\nOptions:")
    print("1. Train a new model (requires labeled dataset)")
    print("2. Download pre-trained model")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        success = train_model()
        if not success:
            logger.info("Training setup completed. Please follow the instructions to add training data.")
    elif choice == '2':
        success = download_pretrained_model()
        if not success:
            logger.info("Please manually download and place a model at models/legal_layout_model.pt")
    else:
        logger.info("Exiting without setting up a model")

if __name__ == "__main__":
    main() 