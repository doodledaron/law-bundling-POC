# Use PaddlePaddle GPU image as base
FROM paddlepaddle/paddle:3.0.0
# FROM paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5
# FROM paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# FROM rapidsai/base:25.06a-cuda11.8-py3.11-amd64

# Keep worker as root for PaddlePaddle compatibility
# PaddlePaddle needs to write to system directories

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

# Python 3.11 for the worker (Pillow >= 13 needs Python >= 3.11; the base image only has 3.10
# and deadsnakes no longer builds 3.11 for Ubuntu 20.04). The base image is kept so poppler
# (pdftoppm) and the system libraries stay identical; only the Python interpreter changes.
# uv installs a pinned python-build-standalone CPython into /opt/venv, which goes first on PATH,
# so `python`, `pip` and `celery` all resolve to 3.11. The base image's 3.10 stays unused.
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python
# The venv is created unseeded and gets only a pinned pip: a seeded setuptools would be overlaid
# (not replaced) by requirements.txt's setuptools under --ignore-installed below.
RUN uv python install 3.11.14 && \
    uv venv --python 3.11.14 /opt/venv && \
    uv pip install --no-cache --python /opt/venv/bin/python pip==26.3
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:${PATH}

# Copy requirements file and the exact-version constraints for this image
COPY requirements.txt constraints-worker.txt ./

# Every pip install below is held to constraints-worker.txt, so a rebuild reproduces the
# tested environment instead of pulling newer transitive packages (e.g. kombu>=5.7 needs redis-py>=6.1).
ARG PIP_CONSTRAINT=/app/constraints-worker.txt

# PaddlePaddle CPU, same version the base image ships for Python 3.10. numpy is pinned here so
# paddle doesn't pull numpy 2.x: the --ignore-installed step below would then lay 1.24 over it
# without removing numpy 2's files, leaving a broken numpy.
RUN pip install --no-cache-dir paddlepaddle==3.0.0 numpy==1.24.3

# Install all requirements from requirements.txt, forcing reinstall of conflicting packages
RUN pip install --no-cache-dir --ignore-installed PyYAML -r requirements.txt
# RUN pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Install PaddleX (but skip hpi-gpu during build - will install at runtime)
# Pinned to the version the running workers use, so a rebuild can't silently change OCR
RUN pip install --no-cache-dir paddlex==3.0.0

# Ensure compatible versions and fix potential conflicts
RUN pip install --no-cache-dir --force-reinstall \
    opencv-python-headless==4.6.0.66 \
    numpy==1.24.3 \
    pillow==13.0.0

# Create necessary directories with proper permissions
RUN mkdir -p uploads results chunks \
    && chmod -R 777 /app/uploads /app/results /app/chunks

# Copy startup script and set permissions
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Set Python path
ENV PYTHONPATH=/app

# Add environment variables for better compatibility
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV OPENCV_IO_MAX_IMAGE_PIXELS=1073741824
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Stay as root for PaddlePaddle compatibility

# Expose port 8000
EXPOSE 8000

# Command to run the application with startup script
CMD ["/app/startup.sh"]
