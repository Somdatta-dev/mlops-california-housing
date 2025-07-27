# CI/CD Pipeline - Quick Reference Guide

## 🚀 Quick Commands

### GitHub CLI Commands

```bash
# View workflow runs
gh run list

# View specific workflow runs
gh run list --workflow=ci.yml
gh run list --workflow=deploy.yml

# View run details
gh run view [RUN_ID]

# Download artifacts
gh run download [RUN_ID]

# Re-run failed jobs
gh run rerun [RUN_ID] --failed

# Trigger manual workflow
gh workflow run workflow-dispatch.yml -f workflow_type=ci
gh workflow run deploy.yml -f environment=staging -f image_tag=v1.0.0
```

### Docker Commands

```bash
# Build images locally
docker build -f Dockerfile -t mlops-housing:gpu .
docker build -f Dockerfile.cpu -t mlops-housing:cpu .

# Test containers
docker run --rm -p 8000:8000 mlops-housing:cpu
docker run --rm --gpus all -p 8000:8000 mlops-housing:gpu

# Check images
docker images | grep mlops-housing
docker inspect mlops-housing:cpu
```

### Deployment Commands

```bash
# SSH to servers
ssh ubuntu@staging.yourdomain.com
ssh ubuntu@production.yourdomain.com

# Check deployment status
docker ps
docker-compose -f docker-compose.staging.yml ps
docker-compose -f docker-compose.production.yml ps

# View logs
docker logs mlops-staging-api
docker logs mlops-prod-api

# Restart services
docker-compose -f docker-compose.staging.yml restart
docker-compose -f docker-compose.production.yml restart
```

## 📊 Monitoring Commands

### Health Checks

```bash
# API health
curl https://staging.yourdomain.com/health
curl https://production.yourdomain.com/health

# Model info
curl https://staging.yourdomain.com/model/info
curl https://production.yourdomain.com/model/info

# Metrics
curl https://staging.yourdomain.com/metrics
curl https://production.yourdomain.com/metrics
```

### Performance Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/locustfile.py --headless -u 50 -r 5 -t 60s --host https://staging.yourdomain.com

# Run specific user class
locust -f tests/locustfile.py --headless -u 10 -r 2 -t 30s --host https://staging.yourdomain.com MLOpsAPIUser
```

## 🔧 Troubleshooting Commands

### Workflow Debugging

```bash
# Validate workflows
python scripts/validate-workflows.py

# Check workflow syntax
yamllint .github/workflows/*.yml

# Test workflow locally (with act)
act -j ci-summary
act -j build-and-push
```

### Server Debugging

```bash
# Check system resources
free -h
df -h
nvidia-smi

# Check Docker
docker system df
docker system prune -f

# Check services
systemctl status docker
systemctl status postgresql
systemctl status nginx
```

### Database Debugging

```bash
# Connect to database
sudo -u postgres psql mlops_staging
sudo -u postgres psql mlops_production

# Check database size
sudo -u postgres psql -c "SELECT pg_size_pretty(pg_database_size('mlops_staging'));"

# Backup database
sudo -u postgres pg_dump mlops_staging > backup_$(date +%Y%m%d).sql
```

## 🔐 Security Commands

### Secret Management

```bash
# List secrets (GitHub CLI)
gh secret list

# Set secret
gh secret set SECRET_NAME

# Delete secret
gh secret delete SECRET_NAME
```

### Security Scanning

```bash
# Run security scans locally
bandit -r src/
safety check
pip-audit

# Docker security scan
docker run --rm -v $(pwd):/app aquasec/trivy fs /app
```

## 📋 Workflow Triggers

### Automatic Triggers

| Event | Workflows Triggered |
|-------|-------------------|
| Push to `main` | CI, Docker Build, Deploy to Staging |
| Push to `develop` | CI, Docker Build |
| Pull Request | PR Checks |
| Tag `v*` | Release, Deploy to Production |
| Daily 2 AM UTC | Dependency Updates |

### Manual Triggers

```bash
# Manual CI run
gh workflow run ci.yml

# Manual deployment
gh workflow run deploy.yml -f environment=staging -f image_tag=latest

# Manual security scan
gh workflow run workflow-dispatch.yml -f workflow_type=security-scan

# Manual performance test
gh workflow run workflow-dispatch.yml -f workflow_type=performance-test -f test_type=performance
```

## 🚨 Emergency Procedures

### Rollback Production

```bash
# Option 1: Via GitHub Actions
gh workflow run deploy.yml -f environment=production -f image_tag=v1.0.0

# Option 2: Manual rollback
ssh ubuntu@production.yourdomain.com
cd /opt/mlops-production
# Find latest backup
ls -la backups/
# Restore configuration
cp backups/docker-compose.production.yml.20240127_120000 docker-compose.production.yml
cp backups/.env.production.20240127_120000 .env.production
# Restart services
docker-compose -f docker-compose.production.yml down
docker-compose -f docker-compose.production.yml up -d
```

### Stop All Workflows

```bash
# Cancel running workflows
gh run list --status in_progress --json databaseId --jq '.[].databaseId' | xargs -I {} gh run cancel {}
```

### Emergency Maintenance Mode

```bash
# Put API in maintenance mode
ssh ubuntu@production.yourdomain.com
docker-compose -f docker-compose.production.yml down mlops-api

# Start maintenance container
docker run -d --name maintenance -p 8000:80 nginx:alpine
docker exec maintenance sh -c 'echo "Service under maintenance" > /usr/share/nginx/html/index.html'
```

## 📈 Performance Monitoring

### Key Metrics to Monitor

```bash
# API Response Time
curl -w "@curl-format.txt" -o /dev/null -s https://production.yourdomain.com/health

# GPU Utilization
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits

# Memory Usage
free -m | awk 'NR==2{printf "Memory Usage: %s/%sMB (%.2f%%)\n", $3,$2,$3*100/$2 }'

# Disk Usage
df -h | awk '$NF=="/"{printf "Disk Usage: %d/%dGB (%s)\n", $3,$2,$5}'
```

### Grafana Queries

```promql
# API Request Rate
rate(http_requests_total[5m])

# API Response Time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# GPU Memory Usage
nvidia_ml_py_memory_used_bytes / nvidia_ml_py_memory_total_bytes * 100

# Container CPU Usage
rate(container_cpu_usage_seconds_total[5m]) * 100
```

## 🔄 Regular Maintenance Tasks

### Daily

```bash
# Check workflow status
gh run list --limit 10

# Check system health
curl https://production.yourdomain.com/health
curl https://staging.yourdomain.com/health

# Review security alerts
gh api repos/:owner/:repo/security-advisories
```

### Weekly

```bash
# Update dependencies (if not automated)
gh workflow run dependency-update.yml

# Review performance metrics
# Check Grafana dashboards

# Clean up old Docker images
docker system prune -f
```

### Monthly

```bash
# Update GitHub Actions versions
# Review and update secrets
# Performance review and optimization
# Security audit and compliance check
```

## 📞 Support Contacts

### Escalation Matrix

| Issue Type | Contact | Response Time |
|------------|---------|---------------|
| Production Down | On-call Engineer | 15 minutes |
| Security Incident | Security Team | 30 minutes |
| Performance Issues | DevOps Team | 1 hour |
| General Issues | Development Team | 4 hours |

### Useful Links

- **Repository**: https://github.com/your-username/mlops-california-housing
- **Staging**: https://staging.yourdomain.com
- **Production**: https://production.yourdomain.com
- **Monitoring**: https://production.yourdomain.com:3000
- **Metrics**: https://production.yourdomain.com:9090

---

*Keep this reference handy for quick access to common CI/CD operations and troubleshooting commands.*