# Install dependencies: pip install fastapi uvicorn jinja2 paddleocr opencv-python numpy pdf2image

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import paddleocr
import cv2
import numpy as np
import re
from pdf2image import convert_from_bytes

# Initialize FastAPI app
app = FastAPI()
# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize PaddleOCR - using en_PP-OCRv3_det (3.8M) model for English text detection
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")

def preprocess_image(img):
    """
    Preprocess the image for better OCR results
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to get rid of the noise
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)
    return denoised

def clean_text(text):
    """
    Clean the extracted text for better regex matching
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters that might interfere with regex
    text = text.replace('\n', ' ').replace('\r', '')
    return text

def extract_nda_fields(text):
    """
    Extract relevant fields from NDA text using flexible regex patterns
    """
    try:
        # More flexible regex patterns that can match different formats
        patterns = {
            # Match company name: Looks for company identifier followed by any company name
            'company': r'(?:(?:Company|Corporation|Corp\.|Inc\.|LLC|Ltd\.)\s*:?\s*)?([A-Za-z0-9\s\.,]+?)(?:\s*\(\"?Disclos(?:er|ing\s*Party)\"|;|,|\n)',
            
            # Match recipient: Looks for recipient identifier followed by name
            'recipient': r'(?:Recipient|Receiving\s*Party)\s*:?\s*([A-Za-z0-9\s\.,]+?)(?:\s*\(\"?Recipient|;|,|\n)',
            
            # Match addresses: Looks for common address patterns
            'company_address': r'(?:place\s*of\s*business|address|located)\s*at\s*([^,;\n]*(?:[0-9]+)[^,;\n]*(?:Road|Street|St\.|Drive|Dr\.|Avenue|Ave\.|Suite|Ste\.|Boulevard|Blvd\.)[^,;\n]*(?:[A-Z]{2})\s*[0-9]{5}(?:-[0-9]{4})?)',
            
            'recipient_address': r'(?:residing|located|with\s*an\s*address)\s*at\s*([^,;\n]*(?:[0-9]+)[^,;\n]*(?:Road|Street|St\.|Drive|Dr\.|Avenue|Ave\.|Suite|Ste\.|Boulevard|Blvd\.)[^,;\n]*(?:[A-Z]{2})\s*[0-9]{5}(?:-[0-9]{4})?)',
            
            # Match duration: Various ways duration might be expressed
            'duration': r'(?:period\s*of|term\s*of|duration\s*of|shall\s*remain\s*in\s*effect\s*for)\s*([^,;\n]*?(?:year|month|day)[^,;\n]*?)(?:from|after|following)',
            
            # Match governing law: Various ways state law might be specified
            'governing_law': r'(?:governed\s*by|pursuant\s*to|under\s*the\s*laws\s*of)\s*(?:the\s*)?(?:State\s*of\s*)?([A-Za-z\s]+)(?:\s*law)?[,\.]',
            
            # Match confidential information: Look for description of what's confidential
            'confidential_info': r'(?:confidential\s*information\s*relating\s*to|concerning|regarding)\s*([^()\n]+?)(?:\s*\(|\.|;)',
            
            # Match dates: Various date formats
            'dates': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b'
        }

        # Extract fields
        fields = {}
        for field, pattern in patterns.items():
            if field == 'dates':
                matches = re.findall(pattern, text, re.IGNORECASE)
                fields[field] = matches if matches else []
            else:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                fields[field] = match.group(1).strip() if match and match.groups() else "Not found"
                # Clean up extra spaces and punctuation
                if fields[field] != "Not found":
                    fields[field] = re.sub(r'\s+', ' ', fields[field]).strip(' .,;')

        return fields

    except Exception as e:
        print(f"Error in pattern matching: {str(e)}")
        return {
            'company': "Not found",
            'recipient': "Not found",
            'company_address': "Not found",
            'recipient_address': "Not found",
            'duration': "Not found",
            'governing_law': "Not found",
            'confidential_info': "Not found",
            'dates': []
        }

# Define root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the main upload page
    """
    return templates.TemplateResponse("index.html", {"request": request})

# Define upload endpoint
@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    """
    Process uploaded file and extract NDA information
    """
    try:
        # Read file contents
        try:
            contents = await file.read()
        except Exception as e:
            return HTMLResponse(content=f"Error reading file: {str(e)}", status_code=500)

        if not contents:
            return HTMLResponse(content="Error: File is empty", status_code=400)

        # Validate file type
        if file.content_type not in ("image/jpeg", "image/png", "application/pdf"):
            return HTMLResponse(
                content="Error: Invalid file type. Only JPEG, PNG, and PDF files are supported.",
                status_code=400,
            )

        # Convert file to image
        try:
            if file.content_type == "application/pdf":
                # Convert PDF to images
                images = convert_from_bytes(contents)
                img = np.array(images[0])  # Use first page
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                # Convert image to OpenCV format
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    return HTMLResponse(content="Error: Could not decode image", status_code=400)
        except Exception as e:
            return HTMLResponse(content=f"Error converting to image: {str(e)}", status_code=500)

        # Preprocess image
        processed_img = preprocess_image(img)

        # Process the image using PaddleOCR
        try:
            result = ocr.ocr(processed_img, cls=True)
        except Exception as e:
            return HTMLResponse(content=f"Error processing OCR: {str(e)}", status_code=500)

        # Concatenate the OCR results into a full text string
        full_text = ""
        for res in result:
            for line in res:
                full_text += line[1][0] + " "

        # Clean the extracted text
        cleaned_text = clean_text(full_text)

        # Extract NDA fields
        fields = extract_nda_fields(cleaned_text)

        # Render template with results
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "full_text": cleaned_text,
                **fields  # Unpack all extracted fields
            },
        )

    except Exception as e:
        return HTMLResponse(content=f"An unexpected error occurred: {str(e)}", status_code=500)

# To run the application: uvicorn main:app --reload