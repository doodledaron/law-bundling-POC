"""
CUDA/GPU Check for PaddleOCR
============================

This script demonstrates how to:
1. Check if CUDA is available on your system
2. Properly initialize PaddleOCR with GPU support
"""

import sys
import os

# Check if PyTorch is available for CUDA checking
try:
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✅ CUDA is available through PyTorch. Found {torch.cuda.device_count()} GPU(s).")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ CUDA is not available through PyTorch. Using CPU for processing (this will be slower).")
except ImportError:
    print("PyTorch not installed. Cannot check CUDA availability.")
    cuda_available = False
    
# Check if PaddlePaddle is available and has GPU support
try:
    import paddle
    paddle.device.set_device('gpu:0')  # Try to use GPU
    if paddle.device.is_compiled_with_cuda():
        print("✅ PaddlePaddle is compiled with CUDA support.")
        print(f"Available devices: {paddle.device.get_available_device()}")
    else:
        print("❌ PaddlePaddle is not compiled with CUDA support.")
except Exception as e:
    print(f"Error checking PaddlePaddle GPU support: {e}")

# Initialize PaddleOCR with GPU support
print("\nTo initialize PaddleOCR with GPU support, use the following code:")
print("""
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    lang='en',
    use_angle_cls=True,
    det_model_dir=None,
    rec_model_dir=None,
    cls_model_dir=None,
    det_limit_side_len=2560,
    det_db_thresh=0.3,
    det_db_box_thresh=0.5,
    rec_batch_num=6,
    rec_char_dict_path=None,
    use_space_char=True,
    use_gpu=True,  # This enables GPU acceleration
    gpu_mem=500,   # Limit GPU memory usage if needed (in MB)
    show_log=False
)
""")

# Try to initialize PaddleOCR with GPU and report the result
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        lang='en',
        use_angle_cls=True,
        use_gpu=True,
        show_log=False
    )
    print("\n✅ Successfully initialized PaddleOCR with GPU support!")
except Exception as e:
    print(f"\n❌ Error initializing PaddleOCR with GPU: {e}")

print("\nTo verify GPU usage during processing:")
print("""
# Add this line before processing to explicitly set the device
import paddle
paddle.device.set_device('gpu:0')
""")

print("\nIMPORTANT: If you're using a Jupyter notebook, add a cell with this code:")
print("""
# Add this to make PaddleOCR use the GPU
%env CUDA_VISIBLE_DEVICES=0
# Or to force CPU mode if you have issues with GPU
# %env CUDA_VISIBLE_DEVICES=
""") 