# Use PaddlePaddle GPU image as base
FROM paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# FROM rapidsai/base:25.06a-cuda11.8-py3.11-amd64

# Ensure we're running as root
USER root

# Set working directory
WORKDIR /app

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

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Create symlink for libcuda.so.1
RUN ln -s /usr/local/cuda/lib64/libcuda.so /usr/lib/libcuda.so.1

# Copy requirements file
COPY requirements.txt .

# Install all requirements from requirements.txt, forcing reinstall of conflicting packages
RUN pip install --no-cache-dir --ignore-installed PyYAML -r requirements.txt
# RUN pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Install PyTorch with CUDA 11.8 support for memory management
RUN pip install --no-cache-dir torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install PaddleX (but skip hpi-gpu during build - will install at runtime)
RUN pip install --no-cache-dir paddlex

# Ensure compatible versions and fix potential conflicts
RUN pip install --no-cache-dir --force-reinstall \
    opencv-python-headless==4.6.0.66 \
    numpy==1.24.3 \
    pillow==10.0.0

# Create necessary directories
RUN mkdir -p uploads results chunks

# Set Python path
ENV PYTHONPATH=/app

# Add environment variables for better compatibility
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV OPENCV_IO_MAX_IMAGE_PIXELS=1073741824
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Copy startup script
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Expose port 8000
EXPOSE 8000

# Command to run the application with startup script
CMD ["/app/startup.sh"]
