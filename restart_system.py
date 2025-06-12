#!/usr/bin/env python3
"""
System Restart Script for Law Document Processing

This script helps restart the system cleanly and fix common issues.
"""
import subprocess
import time
import sys
import redis

def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} - Failed")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {str(e)}")
        return False

def clear_redis():
    """Clear Redis database."""
    print("🗄️  Clearing Redis database...")
    try:
        redis_client = redis.Redis.from_url('redis://localhost:6379/0')
        redis_client.flushdb()
        print("✅ Redis database cleared")
        return True
    except Exception as e:
        print(f"❌ Failed to clear Redis: {str(e)}")
        return False

def main():
    """Main restart sequence."""
    print("🚀 LAW DOCUMENT PROCESSING - SYSTEM RESTART")
    print("=" * 50)
    
    # Step 1: Stop all services
    print("\n📦 STOPPING SERVICES")
    run_command("docker-compose down", "Stopping all Docker services")
    time.sleep(2)
    
    # Step 2: Clean up resources
    print("\n🧹 CLEANING UP")
    run_command("docker system prune -f", "Cleaning Docker resources")
    run_command("docker volume prune -f", "Cleaning Docker volumes")
    
    # Step 3: Start core services
    print("\n🔄 STARTING CORE SERVICES")
    run_command("docker-compose up -d redis", "Starting Redis")
    time.sleep(5)
    
    # Step 4: Clear Redis
    print("\n🗄️  CLEARING DATABASE")
    clear_redis()
    
    # Step 5: Start worker
    print("\n👷 STARTING WORKER")
    run_command("docker-compose up -d worker-documents", "Starting document worker")
    time.sleep(10)
    
    # Step 6: Start API
    print("\n🌐 STARTING API")
    run_command("docker-compose up -d api", "Starting API server")
    time.sleep(5)
    
    # Step 7: Start maintenance services
    print("\n🔧 STARTING MAINTENANCE")
    run_command("docker-compose up -d worker-maintenance beat flower", "Starting maintenance services")
    time.sleep(5)
    
    # Step 8: Check status
    print("\n📊 CHECKING STATUS")
    run_command("docker-compose ps", "Checking container status")
    
    print("\n🎉 System restart complete!")
    print("\n💡 Next steps:")
    print("   1. Wait 30 seconds for services to fully start")
    print("   2. Run: python diagnose_system.py")
    print("   3. Test by uploading a document at http://localhost:8000")
    print("   4. Monitor at http://localhost:5555 (Flower)")

if __name__ == "__main__":
    main() 