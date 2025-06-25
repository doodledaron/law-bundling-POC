#!/usr/bin/env python3
"""
System Diagnostic Script for Law Document Processing

This script helps diagnose issues with Celery, Redis, and task processing.
"""
import redis
import json
import subprocess
import time
from datetime import datetime

def print_header(title):
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def check_redis():
    """Check Redis connectivity and status."""
    print_header("REDIS DIAGNOSTICS")
    
    try:
        # Test Redis connection
        redis_client = redis.Redis.from_url('redis://localhost:6379/0')
        redis_client.ping()
        print("✅ Redis is accessible")
        
        # Get Redis info
        info = redis_client.info()
        print(f"📊 Redis Version: {info.get('redis_version', 'Unknown')}")
        print(f"📈 Memory Used: {info.get('used_memory_human', 'Unknown')}")
        print(f"🔗 Connected Clients: {info.get('connected_clients', 'Unknown')}")
        
        # Check for existing jobs
        job_keys = redis_client.keys("job:*")
        print(f"📋 Active Jobs: {len(job_keys)}")
        
        if job_keys:
            print("🗂️  Recent Jobs:")
            for key in job_keys[:5]:  # Show first 5 jobs
                job_data = redis_client.get(key)
                if job_data:
                    job = json.loads(job_data)
                    job_id = key.decode().split(':')[1][:8]
                    status = job.get('status', 'Unknown')
                    print(f"   {job_id}... - {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis Error: {str(e)}")
        return False

def check_celery():
    """Check Celery worker status."""
    print_header("CELERY DIAGNOSTICS")
    
    try:
        from celery import Celery
        
        # Initialize Celery app
        celery_app = Celery(
            'diagnostics',
            broker='redis://localhost:6379/0',
            backend='redis://localhost:6379/0'
        )
        
        # Get worker status
        i = celery_app.control.inspect()
        
        # Check registered workers
        registered = i.registered()
        if registered:
            print(f"✅ {len(registered)} workers registered:")
            for worker, tasks in registered.items():
                print(f"   🔧 {worker}: {len(tasks)} registered tasks")
        else:
            print("❌ No workers registered")
            return False
        
        # Check active tasks
        active = i.active()
        if active:
            print(f"🔄 Active tasks:")
            for worker, tasks in active.items():
                print(f"   🔧 {worker}: {len(tasks)} active tasks")
                for task in tasks:
                    print(f"      📋 {task.get('name', 'Unknown')}")
        else:
            print("✅ No active tasks (workers idle)")
        
        # Check worker stats
        stats = i.stats()
        if stats:
            print("📊 Worker Statistics:")
            for worker, stat in stats.items():
                pool = stat.get('pool', {})
                processes = pool.get('processes', 'N/A')
                print(f"   🔧 {worker}: {processes} processes")
        
        return True
        
    except Exception as e:
        print(f"❌ Celery Error: {str(e)}")
        return False

def check_docker():
    """Check Docker container status."""
    print_header("DOCKER DIAGNOSTICS")
    
    try:
        # Check container status
        result = subprocess.run(['docker-compose', 'ps'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker Compose status:")
            print(result.stdout)
        else:
            print(f"❌ Docker Compose error: {result.stderr}")
            return False
        
        # Check specific containers
        containers = [
            'law-worker-documents',
            'law-redis',
            'law-api',
            'law-worker-maintenance'
        ]
        
        for container in containers:
            result = subprocess.run(['docker', 'inspect', container], capture_output=True, text=True)
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)[0]
                state = data['State']
                status = state.get('Status', 'Unknown')
                running = state.get('Running', False)
                print(f"   {'✅' if running else '❌'} {container}: {status}")
            else:
                print(f"   ❌ {container}: Not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Docker Error: {str(e)}")
        return False

def test_task_submission():
    """Test submitting a simple task."""
    print_header("TASK SUBMISSION TEST")
    
    try:
        from celery import Celery
        
        # Initialize Celery app
        celery_app = Celery(
            'test',
            broker='redis://localhost:6379/0',
            backend='redis://localhost:6379/0'
        )
        
        # Try to submit warmup task
        print("🧪 Testing warmup task submission...")
        from tasks.ppstructure_tasks import warmup_ppstructure
        
        result = warmup_ppstructure.delay()
        print(f"✅ Task submitted with ID: {result.id}")
        
        # Wait for result
        print("⏳ Waiting for task completion (30s timeout)...")
        try:
            task_result = result.get(timeout=30)
            print(f"✅ Task completed: {task_result}")
            return True
        except Exception as e:
            print(f"⚠️  Task failed or timed out: {str(e)}")
            return False
        
    except Exception as e:
        print(f"❌ Task submission error: {str(e)}")
        return False

def check_logs():
    """Check recent container logs."""
    print_header("RECENT LOGS")
    
    containers = ['law-worker-documents', 'law-redis', 'law-api']
    
    for container in containers:
        print(f"\n📋 Last 10 lines from {container}:")
        try:
            result = subprocess.run(['docker', 'logs', '--tail', '10', container], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:  # Last 10 lines
                    print(f"   {line}")
            else:
                print(f"   ❌ Could not get logs: {result.stderr}")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def main():
    """Run all diagnostics."""
    print(f"🚀 LAW DOCUMENT PROCESSING - SYSTEM DIAGNOSTICS")
    print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'redis': check_redis(),
        'docker': check_docker(),
        'celery': check_celery(),
        'task_test': test_task_submission()
    }
    
    check_logs()
    
    print_header("SUMMARY")
    
    all_good = True
    for component, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {component.upper()}: {'OK' if status else 'FAILED'}")
        if not status:
            all_good = False
    
    if all_good:
        print("\n🎉 All systems operational!")
    else:
        print("\n⚠️  Some issues detected. Check the logs above.")
        
        print("\n💡 Troubleshooting Tips:")
        if not results['redis']:
            print("   - Check if Redis container is running: docker-compose ps")
            print("   - Restart Redis: docker-compose restart redis")
        
        if not results['docker']:
            print("   - Start services: docker-compose up -d")
            print("   - Check logs: docker-compose logs")
        
        if not results['celery']:
            print("   - Restart workers: docker-compose restart worker-documents")
            print("   - Check worker logs: docker-compose logs worker-documents")
        
        if not results['task_test']:
            print("   - Check task routing in celery_config.py")
            print("   - Verify task imports in tasks/__init__.py")

if __name__ == "__main__":
    main() 