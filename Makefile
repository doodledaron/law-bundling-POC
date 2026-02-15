.PHONY: help build up down restart logs logs-api logs-workers logs-redis status clean test rebuild
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo ""
	@echo "  Law Bundling POC - Available Commands"
	@echo "  ======================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# === Build & Run ===

build: ## Build all containers
	docker compose build

up: ## Start all services (detached)
	docker compose up -d

start: build up ## Build and start all services

down: ## Stop all services
	docker compose down

restart: down up ## Restart all services

rebuild: ## Full rebuild (no cache) and start
	docker compose build --no-cache
	docker compose up -d

# === Logs ===

logs: ## Follow all container logs
	docker compose logs -f --tail=100

logs-api: ## Follow API logs only
	docker compose logs -f --tail=100 api

logs-workers: ## Follow all worker logs
	docker compose logs -f --tail=100 worker-documents-container1 worker-documents-container2 worker-documents-container3 worker-documents-container4 worker-documents-container5

N ?= 1
logs-worker: ## Follow single worker log (N=1..5)
	docker compose logs -f --tail=100 worker-documents-container$(N)

logs-redis: ## Follow Redis logs
	docker compose logs -f --tail=100 redis

logs-flower: ## Follow Flower dashboard logs
	docker compose logs -f --tail=100 flower

# === Status & Health ===

status: ## Show running containers
	docker compose ps

health: ## API health check
	curl -s http://localhost:8000/health | python3 -m json.tool || echo "API not responding"

redis-info: ## Redis memory usage
	docker exec law-redis redis-cli INFO memory | grep used_memory_human

disk: ## Check data directory sizes
	docker exec law-worker-documents-container1 du -sh /app/results /app/uploads /app/chunks 2>/dev/null || echo "Container not running"

# === Maintenance ===

clean: ## Clean old data (7d results, 1d chunks)
	docker exec law-worker-documents-container1 find /app/results -maxdepth 1 -type d -mtime +7 -not -path /app/results -exec rm -rf {} +
	docker exec law-worker-documents-container1 find /app/uploads -type f -mtime +7 -delete
	docker exec law-worker-documents-container1 find /app/chunks -type f -mtime +1 -delete
	@echo "Cleanup complete"

nuke: ## Remove all containers, volumes, and networks
	docker compose down -v
	@echo "All containers, volumes, and networks removed"

# === Testing ===

test: ## Run container warmup test
	cd tests && python3 01_summarization_container_warmup_test.py

test-single: ## Run single document upload test
	cd tests && python3 02_summarization_single_upload_test.py

test-relevance: ## Run relevance extraction test (NEW)
	cd tests && python3 04_relevance_extraction_test.py

test-stress: ## Run comprehensive stress test (optional, time-intensive)
	cd tests && python3 03_summarization_stress_test_multipage.py

test-all: test test-single test-relevance ## Run all recommended tests (warmup + single + relevance)

test-full: test-all test-stress ## Run complete test suite (including stress test)

shell: ## Shell into worker container
	docker exec -it law-worker-documents-container1 bash

shell-api: ## Shell into API container
	docker exec -it law-api bash

# === Celery ===

celery-active: ## Show active Celery tasks
	docker exec law-worker-documents-container1 celery -A celery_config inspect active

celery-tasks: ## Show registered Celery tasks
	docker exec law-worker-documents-container1 celery -A celery_config inspect registered

