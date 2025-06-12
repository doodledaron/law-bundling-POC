#!/usr/bin/env python3
"""
Memory monitoring script for Docker containers
"""
import subprocess
import json

def check_container_memory():
    """Check memory usage of all containers."""
    print("🔍 CONTAINER MEMORY USAGE")
    print("=" * 50)
    
    try:
        # Get container stats
        result = subprocess.run(['docker', 'stats', '--no-stream', '--format', 'table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        print(f"Error checking memory: {str(e)}")

def check_system_memory():
    """Check system memory."""
    print("\n🖥️  SYSTEM MEMORY")
    print("=" * 30)
    
    try:
        # Check available memory
        result = subprocess.run(['docker', 'exec', 'law-worker-documents', 'free', '-h'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        print(f"Error checking system memory: {str(e)}")

def check_docker_limits():
    """Check Docker container limits."""
    print("\n📊 CONTAINER LIMITS")
    print("=" * 30)
    
    containers = ['law-worker-documents', 'law-api', 'law-redis']
    
    for container in containers:
        try:
            result = subprocess.run(['docker', 'inspect', container], capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)[0]
                host_config = data.get('HostConfig', {})
                memory = host_config.get('Memory', 0)
                memory_swap = host_config.get('MemorySwap', 0)
                
                memory_gb = memory / (1024**3) if memory > 0 else "unlimited"
                swap_gb = memory_swap / (1024**3) if memory_swap > 0 else "unlimited"
                
                print(f"{container}:")
                print(f"  Memory Limit: {memory_gb}GB")
                print(f"  Memory+Swap:  {swap_gb}GB")
                print()
                
        except Exception as e:
            print(f"Error checking {container}: {str(e)}")

if __name__ == "__main__":
    check_container_memory()
    check_system_memory()
    check_docker_limits() 