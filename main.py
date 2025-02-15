# Install dependencies: pip install fastapi uvicorn jinja2 paddleocr opencv-python numpy pdf2image

from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import paddleocr
import cv2
import numpy as np
import re
from pdf2image import convert_from_bytes
from transformers import pipeline

def generate_summary(text):
    """
    Generate a summary of the input text using the transformers library.
    Using sshleifer/distilbart-cnn-12-6 for better performance on shorter texts.
    """
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    # Limit input text to 1024 tokens to prevent model overload
    text = ' '.join(text.split()[:1024])
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False, truncation=True)
    return summary[0]['summary_text']

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize PaddleOCR - using en_PP-OCRv3_det (3.8M) model for English text detection
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")

# Error messages
ERROR_MESSAGES = {
    "file_required": "No file was uploaded. Please select a file.",
    "invalid_type": "Only PDF, JPEG, and PNG files are supported.",
    "empty_file": "The uploaded file is empty.",
    "ocr_error": "Error processing the document. Please try again.",
    "decode_error": "Could not decode the image. Please try another file."
}

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
        patterns = {
            # Match company name - fixed to remove trailing colon
            'company': r'between:\s+(.*?)(?=\s*:?\s*\("Discloser")',

            # Match recipient name - improved to get just the name
            'recipient': r'and\.\s+(.*?)(?=\s*:\s*\("Recipient")',

            # Match company address - unchanged as it works correctly
            'company_address': r'(?:business\s*at\s*)(.*?)(?:;)',

            # Match recipient address - unchanged as it works correctly
            'recipient_address': r'(?:residing\s*at\s*)(.*?)(?:\.)',

            # Match both initial duration and survival period
            'duration': r'period\s+of\s+(.*?)\s+years.*?additional\s+(.*?)\s+years',

            # Match governing law - fixed to capture full state law reference
            'governing_law': r'governed by and construed in accordance with the laws of the\.?\s*([^.]+?)(?:\.|$)',

            # Match confidential information - improved to capture full scope
            'confidential_info': r'information\s+relating\s+to\s+(.*?)(?=\s*\(the "Confidential Information"\))',

            # Match dates - improved format handling
            'dates': r'\b(?:February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b'
        }

        # Extract fields
        fields = {}
        for field, pattern in patterns.items():
            if field == 'dates':
                matches = re.findall(pattern, text, re.IGNORECASE)
                fields[field] = list(set(matches)) if matches else []  # Remove duplicates
            elif field == 'duration':
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match and match.groups():
                    initial_term = match.group(1).strip()
                    survival_period = match.group(2).strip()
                    fields[field] = f"{initial_term} years with {survival_period} years survival period"
                else:
                    fields[field] = "Not found"
            else:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                fields[field] = match.group(1).strip() if match and match.groups() else "Not found"
                # Clean up extra spaces and punctuation
                if fields[field] != "Not found":
                    fields[field] = re.sub(r'\s+', ' ', fields[field]).strip(' .,;:')  # Added ':' to strip

        # Post-processing for governing law to ensure complete phrase
        if fields['governing_law'] != "Not found":
            fields['governing_law'] = "laws of the " + fields['governing_law']

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
            error_message = f"Error reading file: {str(e)}"
            print(error_message)  # Log the error
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": error_message},
                status_code=500
            )

        # Check file size (limit to 10MB)
        if len(contents) > 10 * 1024 * 1024:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "File size exceeds 10MB limit."},
                status_code=400
            )

        if not contents:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": ERROR_MESSAGES["empty_file"]},
                status_code=400
            )

        # Validate file type
        if file.content_type not in ("image/jpeg", "image/png", "application/pdf"):
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": ERROR_MESSAGES["invalid_type"]},
                status_code=400
            )

        # Convert file to image
        try:
            if file.content_type == "application/pdf":
                # Convert PDF to images
                images = convert_from_bytes(contents)
                full_text = ""
                confidence_scores = []
                
                for image in images:
                    img = np.array(image)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    # Preprocess image
                    processed_img = preprocess_image(img)

                    # Process the image using PaddleOCR
                    try:
                        result = ocr.ocr(processed_img, cls=True)
                    except Exception as e:
                        error_message = f"OCR processing error: {str(e)}"
                        print(error_message)  # Log the error
                        return templates.TemplateResponse(
                            "index.html",
                            {"request": request, "error": error_message},
                            status_code=500
                        )

                    # Concatenate the OCR results into a full text string
                    for res in result:
                        for line in res:
                            full_text += line[1][0] + " "
                            confidence_scores.append(line[1][1])
                
                cleaned_text = clean_text(full_text)
                if confidence_scores:
                    average_confidence = sum(confidence_scores) / len(confidence_scores)
                    average_confidence_formatted = f"{average_confidence:.2%}"
                else:
                    average_confidence_formatted = "N/A"
            else:
                # Convert image to OpenCV format
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    error_message = f"Image decode error: Could not decode the image. Please try another file."
                    print(error_message)  # Log the error
                    return templates.TemplateResponse(
                        "index.html",
                        {"request": request, "error":  error_message},
                        status_code=400
                    )

                # Process single image
                processed_img = preprocess_image(img)
                try:
                    result = ocr.ocr(processed_img, cls=True)
                except Exception as e:
                    return templates.TemplateResponse(
                        "index.html",
                        {"request": request, "error": ERROR_MESSAGES["ocr_error"]},
                        status_code=500
                    )

                # Extract text and confidence scores
                full_text = ""
                confidence_scores = []
                for res in result:
                    for line in res:
                        full_text += line[1][0] + " "
                        confidence_scores.append(line[1][1])
                
                cleaned_text = clean_text(full_text)
                if confidence_scores:
                    average_confidence = sum(confidence_scores) / len(confidence_scores)
                    average_confidence_formatted = f"{average_confidence:.2%}"
                else:
                    average_confidence_formatted = "N/A"

        except Exception as e:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": f"Error processing the document: {str(e)}"},
                status_code=500
            )

        # Extract NDA fields
        fields = extract_nda_fields(cleaned_text)

        # Generate summary
        summary = generate_summary(cleaned_text)

        # Render template with results
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "full_text": cleaned_text,
                "average_confidence_formatted": average_confidence_formatted,
                "summary": summary,
                **fields
            },
        )

    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(error_message)  # Log the error
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": error_message},
            status_code=500
        )
