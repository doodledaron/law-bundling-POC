#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import importlib
import subprocess
import platform
from pathlib import Path

def check_package(package_name):
    """Check if a package is installed and return its version."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown version")
        return True, version
    except ImportError:
        return False, None

def check_cuda():
    """Check if CUDA is available through PyTorch."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        device_count = torch.cuda.device_count() if cuda_available else 0
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)] if cuda_available else []
        return {
            "available": cuda_available,
            "version": cuda_version,
            "device_count": device_count,
            "devices": device_names
        }
    except ImportError:
        return {
            "available": False,
            "version": None,
            "device_count": 0,
            "devices": []
        }

def check_disk_space(path):
    """Check available disk space."""
    try:
        if platform.system() == 'Windows':
            total, used, free = shutil.disk_usage(path)
        else:
            stat = os.statvfs(path)
            free = stat.f_frsize * stat.f_bavail
            total = stat.f_frsize * stat.f_blocks
        
        # Convert to GB
        total_gb = total / (1024**3)
        free_gb = free / (1024**3)
        return {
            "total_gb": round(total_gb, 2),
            "free_gb": round(free_gb, 2),
            "percent_free": round((free / total) * 100, 2)
        }
    except:
        return None

def check_pdf_library():
    """Check if PDF libraries are working."""
    try:
        from pdf2image import convert_from_path
        import PIL
        return True
    except Exception as e:
        return False

def check_dataset_files():
    """Check if the dataset files exist."""
    dataset_dir = Path("CUAD_v1/layoutlmv3_dataset")
    return {
        "exists": dataset_dir.exists(),
        "train": (dataset_dir / "train").exists() if dataset_dir.exists() else False,
        "val": (dataset_dir / "val").exists() if dataset_dir.exists() else False,
        "test": (dataset_dir / "test").exists() if dataset_dir.exists() else False
    }

def check_model_files():
    """Check if model files exist."""
    model_dir = Path("layoutlmv3/model")
    best_model_dir = model_dir / "best_model"
    return {
        "exists": model_dir.exists(),
        "best_model": best_model_dir.exists() if model_dir.exists() else False,
        "has_config": (best_model_dir / "config.json").exists() if best_model_dir.exists() else False,
        "has_model": (best_model_dir / "pytorch_model.bin").exists() if best_model_dir.exists() else False
    }

def check_transformers_cache():
    """Check the Hugging Face transformers cache directory."""
    try:
        from transformers.utils import TRANSFORMERS_CACHE
        cache_dir = Path(TRANSFORMERS_CACHE)
        return {
            "path": str(cache_dir),
            "exists": cache_dir.exists(),
            "writable": os.access(cache_dir, os.W_OK) if cache_dir.exists() else False
        }
    except:
        home_dir = Path.home()
        cache_dir = home_dir / ".cache" / "huggingface" / "transformers"
        return {
            "path": str(cache_dir),
            "exists": cache_dir.exists(),
            "writable": os.access(cache_dir, os.W_OK) if cache_dir.exists() else False
        }

def main():
    """Run all checks and print results."""
    import json
    import shutil
    
    print("=" * 60)
    print("LayoutLMv3 Environment Diagnostics")
    print("=" * 60)
    
    print("\n## System Information:")
    print(f"- Operating System: {platform.system()} {platform.release()}")
    print(f"- Python Version: {platform.python_version()}")
    print(f"- CPU: {platform.processor() or 'Unknown'}")
    
    print("\n## Required Packages:")
    required_packages = [
        "torch", "transformers", "pdf2image", "PIL", "matplotlib", 
        "opencv-python", "tqdm", "scikit-learn", "pandas", "paddlepaddle", "paddleocr"
    ]
    
    missing_packages = []
    for package in required_packages:
        installed, version = check_package(package.split("-")[0])  # Handle packages like opencv-python
        status = "✓" if installed else "✗"
        version_str = f"v{version}" if installed else "Not installed"
        print(f"- {package:<15} {status} {version_str}")
        if not installed:
            missing_packages.append(package)
    
    print("\n## CUDA Information:")
    cuda_info = check_cuda()
    cuda_status = "✓" if cuda_info["available"] else "✗"
    print(f"- CUDA available: {cuda_status}")
    if cuda_info["available"]:
        print(f"- CUDA version: {cuda_info['version']}")
        print(f"- GPU devices: {cuda_info['device_count']}")
        for i, device in enumerate(cuda_info['devices']):
            print(f"  - GPU {i}: {device}")
    
    print("\n## Storage:")
    space = check_disk_space(".")
    if space:
        print(f"- Total disk space: {space['total_gb']:.2f} GB")
        print(f"- Free disk space: {space['free_gb']:.2f} GB ({space['percent_free']}%)")
        if space['free_gb'] < 10:
            print("  ⚠️ Warning: Less than 10GB free disk space")
    
    print("\n## Dataset Status:")
    dataset_status = check_dataset_files()
    print(f"- Dataset directory exists: {'✓' if dataset_status['exists'] else '✗'}")
    print(f"- Train dataset: {'✓' if dataset_status['train'] else '✗'}")
    print(f"- Validation dataset: {'✓' if dataset_status['val'] else '✗'}")
    print(f"- Test dataset: {'✓' if dataset_status['test'] else '✗'}")
    
    print("\n## Model Status:")
    model_status = check_model_files()
    print(f"- Model directory exists: {'✓' if model_status['exists'] else '✗'}")
    print(f"- Best model directory: {'✓' if model_status['best_model'] else '✗'}")
    print(f"- Model config: {'✓' if model_status['has_config'] else '✗'}")
    print(f"- Model weights: {'✓' if model_status['has_model'] else '✗'}")
    
    print("\n## Transformers Cache:")
    cache_status = check_transformers_cache()
    print(f"- Cache directory: {cache_status['path']}")
    print(f"- Cache exists: {'✓' if cache_status['exists'] else '✗'}")
    print(f"- Cache writable: {'✓' if cache_status['writable'] else '✗'}")
    
    print("\n## PDF Processing:")
    pdf_ok = check_pdf_library()
    print(f"- PDF libraries: {'✓' if pdf_ok else '✗'}")
    
    print("\n## Summary:")
    if missing_packages:
        print(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install " + " ".join(missing_packages))
    else:
        print("✓ All required packages installed.")
    
    if not cuda_info["available"]:
        print("⚠️ CUDA not available - training will be slow on CPU.")
    
    if not dataset_status["exists"]:
        print("⚠️ Dataset not prepared. Run the prepare_dataset.py script first.")
    
    if not model_status["has_model"]:
        print("⚠️ Model not trained. Run the train.py script or download a pre-trained model.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 