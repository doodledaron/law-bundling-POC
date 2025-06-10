# Use PaddlePaddle GPU image as base
# FROM paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM rapidsai/base:25.06a-cuda11.8-py3.11-amd64

# Ensure we're running as root
USER root

# Set working directory
WORKDIR /app

# Install Python 3.11 and pip (commented out since base image already has Python 3.11)
# RUN apt-get update && \
#     apt-get install -y \
#         software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update && \
#     apt-get install -y \
#         python3.11 \
#         python3.11-distutils \
#         python3.11-venv \
#         python3-pip && \
#     ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
#     python3 -m pip install --upgrade pip

# Install system dependencies and ccache (optional, suppresses warning)
RUN apt-get update && \
    apt-get install -y \
        poppler-utils \
        libpoppler-cpp-dev \
        curl \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        ccache \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install all requirements from requirements.txt, forcing reinstall of conflicting packages
RUN pip install --no-cache-dir --ignore-installed PyYAML -r requirements.txt
RUN pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Ensure compatible versions and fix potential conflicts
RUN pip install --no-cache-dir --force-reinstall \
    opencv-python-headless==4.6.0.66 \
    numpy==1.24.3 \
    pillow==10.0.0

# Create necessary directories
RUN mkdir -p uploads results chunks static

# Set Python path
ENV PYTHONPATH=/app

# Add environment variables for better compatibility
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV OPENCV_IO_MAX_IMAGE_PIXELS=1073741824
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Expose port 8000
EXPOSE 8000

# Command to run the application (will be overridden in docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
