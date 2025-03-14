FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for pdf2image, PyMuPDF, OpenCV, PaddleOCR, and SciPy
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libmupdf-dev \
    gcc \
    g++ \
    libglib2.0-dev \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    ffmpeg \
    wget \
    libatlas-base-dev \
    gfortran \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt  

COPY . .

# Create necessary directories
RUN mkdir -p uploads results chunks

# Changed port to match main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
