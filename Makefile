# Makefile for MLOps California Housing Platform Docker Management
# Provides convenient commands for building, running, and managing containers

.PHONY: help build build-dev build-prod run run-dev run-prod stop clean logs test gpu-test

# Default target
help: ## Show this help message
	@echo "MLOps California Housing Platform - Docker Management"
	@echo "======================================================"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Build targets
build: ## Build production Docker image
	@echo "Building production Docker image..."
	./docker/build.sh latest production

build-dev: ## Build development Docker image
	@echo "Building development Docker image..."
	./docker/build.sh latest development

build-cpu: ## Build CPU-only Docker image
	@echo "Building CPU-only Docker image..."
	docker build -f Dockerfile.cpu -t mlops-california-housing:cpu --target production .

build-prod: ## Build production Docker image with optimizations
	@echo "Building optimized production Docker image..."
	NO_CACHE=true ./docker/build.sh latest production

build-all: ## Build both development and production images
	@echo "Building all Docker images..."
	./docker/build.sh latest development
	./docker/build.sh latest production
	$(MAKE) build-cpu

# Run targets
run: ## Start all services with Docker Compose
	@echo "Starting all services..."
	docker-compose up -d

run-dev: ## Start development environment
	@echo "Starting development environment..."
	docker-compose --profile development up -d

run-cpu: ## Start CPU-only environment
	@echo "Starting CPU-only environment..."
	docker-compose --profile cpu-only up -d

run-prod: ## Start production environment
	@echo "Starting production environment..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

run-single: ## Run single API container for testing
	@echo "Running single API container..."
	docker run -d --name mlops-test --gpus all -p 8000:8000 mlops-california-housing:latest

run-single-cpu: ## Run single CPU-only API container for testing
	@echo "Running single CPU-only API container..."
	docker run -d --name mlops-test-cpu -p 8003:8000 mlops-california-housing:cpu

# Management targets
stop: ## Stop all running containers
	@echo "Stopping all containers..."
	docker-compose down
	docker-compose --profile development down 2>/dev/null || true
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml down 2>/dev/null || true

restart: ## Restart all services
	@echo "Restarting all services..."
	$(MAKE) stop
	$(MAKE) run

clean: ## Clean up containers, images, and volumes
	@echo "Cleaning up Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

clean-all: ## Clean up everything including images
	@echo "Cleaning up all Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -af
	docker volume prune -f

# Monitoring targets
logs: ## Show logs from all services
	docker-compose logs -f

logs-api: ## Show API logs only
	docker-compose logs -f mlops-api

logs-mlflow: ## Show MLflow logs only
	docker-compose logs -f mlflow

status: ## Show status of all containers
	@echo "Container Status:"
	@echo "=================="
	docker-compose ps
	@echo ""
	@echo "Resource Usage:"
	@echo "==============="
	docker stats --no-stream

health: ## Check health of all services
	@echo "Health Check Results:"
	@echo "===================="
	@curl -s http://localhost:8000/health | jq . || echo "API: Not responding"
	@curl -s http://localhost:5000/health | jq . || echo "MLflow: Not responding"
	@curl -s http://localhost:9090/-/healthy || echo "Prometheus: Not responding"
	@curl -s http://localhost:3000/api/health | jq . || echo "Grafana: Not responding"

# Testing targets
test: ## Run tests in container
	@echo "Running tests in container..."
	docker-compose exec mlops-api python -m pytest tests/ -v

test-api: ## Test API endpoints
	@echo "Testing API endpoints..."
	@curl -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"MedInc":8.3252,"HouseAge":41.0,"AveRooms":6.984,"AveBedrms":1.024,"Population":322.0,"AveOccup":2.556,"Latitude":37.88,"Longitude":-122.23}' \
		| jq .

gpu-test: ## Test GPU availability in container
	@echo "Testing GPU availability..."
	docker-compose exec mlops-api nvidia-smi
	docker-compose exec mlops-api python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Development targets
shell: ## Open shell in API container
	docker-compose exec mlops-api /bin/bash

shell-root: ## Open root shell in API container
	docker-compose exec --user root mlops-api /bin/bash

jupyter: ## Start Jupyter notebook server
	@echo "Starting Jupyter notebook server..."
	docker-compose --profile development up -d jupyter
	@echo "Jupyter available at: http://localhost:8888"

# Database targets
db-shell: ## Open database shell
	docker-compose exec mlops-api sqlite3 /app/data/mlops_platform.db

db-backup: ## Backup database
	@echo "Creating database backup..."
	docker-compose exec mlops-api sqlite3 /app/data/mlops_platform.db ".backup /app/data/backup_$(shell date +%Y%m%d_%H%M%S).db"

# Monitoring targets
metrics: ## Show Prometheus metrics
	@curl -s http://localhost:8000/metrics

grafana-reset: ## Reset Grafana admin password
	docker-compose exec grafana grafana-cli admin reset-admin-password admin123

# Deployment targets
deploy-staging: ## Deploy to staging environment
	@echo "Deploying to staging..."
	docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

deploy-prod: ## Deploy to production environment
	@echo "Deploying to production..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scaling targets
scale-api: ## Scale API service (usage: make scale-api REPLICAS=3)
	@echo "Scaling API service to $(REPLICAS) replicas..."
	docker-compose up -d --scale mlops-api=$(REPLICAS)

# Maintenance targets
update: ## Update all images and rebuild
	@echo "Updating images and rebuilding..."
	docker-compose pull
	$(MAKE) build-all
	$(MAKE) restart

backup: ## Create full backup of volumes and data
	@echo "Creating full backup..."
	mkdir -p backups
	docker run --rm -v mlflow_data:/data -v $(PWD)/backups:/backup alpine tar czf /backup/mlflow_$(shell date +%Y%m%d_%H%M%S).tar.gz /data
	docker run --rm -v prometheus_data:/data -v $(PWD)/backups:/backup alpine tar czf /backup/prometheus_$(shell date +%Y%m%d_%H%M%S).tar.gz /data
	docker run --rm -v grafana_data:/data -v $(PWD)/backups:/backup alpine tar czf /backup/grafana_$(shell date +%Y%m%d_%H%M%S).tar.gz /data

# Security targets
security-scan: ## Scan images for vulnerabilities
	@echo "Scanning images for security vulnerabilities..."
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image mlops-california-housing:latest

# Configuration targets
config: ## Show Docker Compose configuration
	docker-compose config

config-prod: ## Show production Docker Compose configuration
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml config

# Environment setup
setup-env: ## Create .env file from template
	@if [ ! -f .env ]; then \
		cp .env.template .env; \
		echo ".env file created from template. Please edit it with your configuration."; \
	else \
		echo ".env file already exists."; \
	fi

# Default values for variables
REPLICAS ?= 2