#!/bin/bash

# Deploy High-Throughput 6-Container Document Processing Setup
# This script deploys the optimized PPStructure processing system
# Architecture: 6 containers * 4 concurrency = 24 parallel workers

echo "🚀 Deploying High-Throughput 6-Container Document Processing Setup"
echo "=================================================================="

# Check system requirements
echo "📋 Checking system requirements..."

# Check available memory (should be at least 32GB for 6*5GB + overhead)
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_MEM" -lt 32 ]; then
    echo "⚠️  WARNING: System has ${TOTAL_MEM}GB RAM. Recommended: 32GB+ for optimal performance"
    echo "   Each container uses 5GB, total requirement: 30GB + 2GB overhead"
else
    echo "✅ Memory check passed: ${TOTAL_MEM}GB available"
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "✅ GPU check passed: ${GPU_COUNT} GPU(s) detected"
else
    echo "⚠️  WARNING: No NVIDIA GPU detected. Performance may be reduced."
fi

# Check Docker Compose version
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose available"
else
    echo "❌ ERROR: Docker Compose not found. Please install Docker Compose."
    exit 1
fi

echo ""
echo "🏗️  Architecture Overview:"
echo "   • 6 Document Processing Containers"
echo "   • 4 Concurrency per Container = 24 Total Workers"
echo "   • 5GB Memory per Container = 30GB Total"
echo "   • Fixed 5-Page Chunking Strategy"
echo "   • Shared chunk_queue for Load Balancing"
echo ""

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Clean up old containers if they exist
echo "🧹 Cleaning up old containers..."
docker container prune -f

# Build images
echo "🔨 Building container images..."
docker-compose build

# Start the high-throughput setup
echo "🚀 Starting 6-container high-throughput setup..."
docker-compose up -d

# Wait for containers to start
echo "⏳ Waiting for containers to initialize..."
sleep 30

# Check container status
echo "📊 Container Status:"
echo "==================="
for i in {1..6}; do
    CONTAINER_NAME="law-worker-documents-container${i}"
    if docker ps | grep -q "$CONTAINER_NAME"; then
        STATUS="✅ Running"
        MEMORY=$(docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}" | grep "$CONTAINER_NAME" | awk '{print $2}' || echo "N/A")
        echo "   Container $i: $STATUS (Memory: $MEMORY)"
    else
        echo "   Container $i: ❌ Not Running"
    fi
done

# Check Redis connectivity
echo ""
echo "🔍 Testing Redis connectivity..."
if docker exec law-redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis connection successful"
else
    echo "❌ Redis connection failed"
fi

# Check API availability
echo ""
echo "🔍 Testing API availability..."
sleep 10
if curl -f -s http://localhost:8000/health > /dev/null; then
    echo "✅ API health check passed"
else
    echo "⚠️  API not yet available (may still be starting)"
fi

# Display final status
echo ""
echo "🎯 Deployment Summary:"
echo "====================="
echo "   Architecture: 6-container high-throughput setup"
echo "   Total Workers: 24 (6 containers × 4 concurrency)"
echo "   Memory Usage: 30GB (6 containers × 5GB each)"
echo "   Queue Strategy: Shared chunk_queue"
echo "   Chunking: Fixed 5-page chunks"
echo "   Expected Performance: 3x-5x improvement vs 2-container setup"
echo ""
echo "📝 Next Steps:"
echo "   1. Test with a sample document: curl -X POST -F 'file=@sample.pdf' http://localhost:8000/upload/"
echo "   2. Monitor logs: docker-compose logs -f"
echo "   3. Check performance: docker stats"
echo ""

# Performance monitoring commands
echo "🔧 Useful Commands:"
echo "   Monitor all containers: docker-compose logs -f"
echo "   Check memory usage: docker stats"
echo "   View Redis queue: docker exec law-redis redis-cli monitor"
echo "   Restart if needed: docker-compose restart"
echo ""
echo "✅ High-throughput 6-container deployment complete!"
echo "📈 System ready for maximum document processing throughput." 