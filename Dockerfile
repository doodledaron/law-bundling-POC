FROM python:3.11-slim

WORKDIR /app

# Set Python environment variables for better memory management
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Limit math library threads to reduce memory usage
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    # Set timezone
    TZ=UTC

# Install system dependencies for pdf2image
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # Make sure psutil is installed for memory monitoring 
    && pip install --no-cache-dir psutil==5.9.5 \
    # Clean pip cache
    && rm -rf ~/.cache/pip

COPY . .

# Add a healthcheck to ensure the service is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# Default command (will be overridden by docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]