#!/bin/bash

echo "Starting Law Document Processing Application..."

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, installing high-performance inference plugin..."
    
    # Create symlink for libcuda.so.1 if it doesn't exist
    if [ ! -f /usr/lib/libcuda.so.1 ]; then
        if [ -f /usr/local/cuda/lib64/libcuda.so ]; then
            ln -s /usr/local/cuda/lib64/libcuda.so /usr/lib/libcuda.so.1
            echo "Created symlink for libcuda.so.1"
        fi
    fi
    
    # Try to install hpi-gpu plugin
    echo "Installing high-performance inference plugin for GPU..."
    paddlex --install hpi-gpu || echo "Warning: Failed to install high-performance inference plugin, continuing without it"
    
    # Verify CUDA setup
    echo "Verifying CUDA setup..."
    nvidia-smi
    python3 -c "import paddle; print('PaddlePaddle version:', paddle.__version__); print('CUDA available:', paddle.device.is_compiled_with_cuda())"
    
    # Verify PyTorch CUDA setup for memory management
    echo "Verifying PyTorch CUDA setup..."
    python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device count:', torch.cuda.device_count())
    print('Current CUDA device:', torch.cuda.current_device())
    print('CUDA device name:', torch.cuda.get_device_name(0))
else:
    print('Warning: PyTorch CUDA not available - memory management features will be limited')
"
else
    echo "No GPU detected, running in CPU mode"
fi

echo "Starting FastAPI application..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 