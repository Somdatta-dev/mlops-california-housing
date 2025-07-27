# GitHub Actions CI/CD Pipeline - Complete Setup Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Initial Setup](#initial-setup)
4. [Secrets Configuration](#secrets-configuration)
5. [Environment Setup](#environment-setup)
6. [Workflow Configuration](#workflow-configuration)
7. [Testing the Pipeline](#testing-the-pipeline)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## 🎯 Overview

This CI/CD pipeline provides enterprise-grade automation for the MLOps California Housing Platform, including:

- **Continuous Integration**: Automated testing, code quality checks, and security scanning
- **Continuous Deployment**: Automated deployment to staging and production environments
- **Docker Management**: Multi-architecture image building and registry management
- **Security Monitoring**: Vulnerability scanning and dependency updates
- **Performance Testing**: Load testing and performance benchmarking

### Pipeline Architecture

```mermaid
graph TB
    A[Code Push/PR] --> B[CI Workflow]
    B --> C[Code Quality]
    B --> D[Unit Tests]
    B --> E[Security Scan]
    B --> F[Docker Build Test]
    
    G[Main Branch Push] --> H[Docker Build & Push]
    H --> I[Deploy to Staging]
    I --> J[Staging Tests]
    
    K[Release Tag] --> L[Release Workflow]
    L --> M[Build Release Images]
    M --> N[Deploy to Production]
    N --> O[Production Tests]
    
    P[Daily Schedule] --> Q[Dependency Updates]
    Q --> R[Security Monitoring]
```

## 🔧 Prerequisites

### Required Tools and Accounts

1. **GitHub Repository** with Actions enabled
2. **Docker Hub Account** (optional, for multi-registry support)
3. **Staging Server** with Docker and SSH access
4. **Production Server** with Docker and SSH access
5. **Slack Workspace** (optional, for notifications)

### Server Requirements

#### Staging Server
- **OS**: Ubuntu 20.04+ or similar Linux distribution
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB minimum, 100GB recommended
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **GPU**: NVIDIA GPU with CUDA support (optional)

#### Production Server
- **OS**: Ubuntu 20.04+ or similar Linux distribution
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 100GB minimum, 200GB recommended
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **GPU**: NVIDIA GPU with CUDA support (recommended)

## 🚀 Initial Setup

### Step 1: Repository Setup

1. **Fork or Clone the Repository**
   ```bash
   git clone https://github.com/your-username/mlops-california-housing.git
   cd mlops-california-housing
   ```

2. **Verify Workflow Files**
   ```bash
   # Check if workflow files exist
   ls -la .github/workflows/
   
   # Validate workflows (optional)
   python scripts/validate-workflows.py
   ```

3. **Enable GitHub Actions**
   - Go to your repository on GitHub
   - Navigate to **Settings** → **Actions** → **General**
   - Ensure "Allow all actions and reusable workflows" is selected

### Step 2: Server Preparation

#### Staging Server Setup

1. **Install Docker and Docker Compose**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Add user to docker group
   sudo usermod -aG docker $USER
   
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   
   # Verify installation
   docker --version
   docker-compose --version
   ```

2. **Setup NVIDIA Docker (for GPU support)**
   ```bash
   # Install NVIDIA Docker
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   
   # Test GPU access
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

3. **Create Deployment Directory**
   ```bash
   sudo mkdir -p /opt/mlops-staging
   sudo chown $USER:$USER /opt/mlops-staging
   cd /opt/mlops-staging
   
   # Create required directories
   mkdir -p data models logs plots backups
   ```

4. **Setup SSH Key Authentication**
   ```bash
   # Generate SSH key pair (if not exists)
   ssh-keygen -t rsa -b 4096 -C "github-actions@yourdomain.com"
   
   # Add public key to authorized_keys
   cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   
   # Copy private key for GitHub Secrets (save this securely)
   cat ~/.ssh/id_rsa
   ```

#### Production Server Setup

Repeat the same steps as staging server setup, but use `/opt/mlops-production` as the deployment directory.

## 🔐 Secrets Configuration

### Step 3: GitHub Secrets Setup

Navigate to your GitHub repository → **Settings** → **Secrets and variables** → **Actions**

#### Required Secrets

1. **Docker Registry Secrets**
   ```
   DOCKERHUB_USERNAME=your-dockerhub-username
   DOCKERHUB_TOKEN=your-dockerhub-access-token
   ```

2. **Staging Server Secrets**
   ```
   STAGING_HOST=staging.yourdomain.com
   STAGING_USER=ubuntu
   STAGING_SSH_KEY=-----BEGIN RSA PRIVATE KEY-----
   [Your staging server private key]
   -----END RSA PRIVATE KEY-----
   STAGING_PORT=22
   ```

3. **Production Server Secrets**
   ```
   PROD_HOST=production.yourdomain.com
   PROD_USER=ubuntu
   PROD_SSH_KEY=-----BEGIN RSA PRIVATE KEY-----
   [Your production server private key]
   -----END RSA PRIVATE KEY-----
   PROD_PORT=22
   ```

4. **Database Secrets**
   ```
   STAGING_DB_HOST=staging-db.yourdomain.com
   STAGING_DB_USER=mlops_user
   STAGING_DB_PASSWORD=secure_staging_password
   STAGING_DB_NAME=mlops_staging
   
   PROD_DB_HOST=prod-db.yourdomain.com
   PROD_DB_USER=mlops_user
   PROD_DB_PASSWORD=secure_production_password
   PROD_DB_NAME=mlops_production
   ```

5. **API Keys**
   ```
   STAGING_API_KEY=staging-api-key-here
   PROD_API_KEY=production-api-key-here
   ```

6. **Monitoring Secrets**
   ```
   GRAFANA_ADMIN_PASSWORD=secure_grafana_password
   ```

7. **DVC Configuration**
   ```
   DVC_REMOTE_URL=gdrive://your-google-drive-folder-id
   ```

8. **Notification Secrets**
   ```
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
   ```

### Step 4: Environment Configuration

#### GitHub Environments Setup

1. **Create Staging Environment**
   - Go to **Settings** → **Environments**
   - Click **New environment**
   - Name: `staging`
   - Protection rules: None (for automatic deployment)
   - Environment secrets: Add staging-specific secrets

2. **Create Production Environment**
   - Name: `production`
   - Protection rules:
     - ✅ Required reviewers (add team members)
     - ✅ Restrict pushes to protected branches
     - ✅ Deployment branches: `main` and tags matching `v*`
   - Environment secrets: Add production-specific secrets

## ⚙️ Workflow Configuration

### Step 5: Customize Workflow Settings

#### Update Registry Configuration

Edit `.github/workflows/docker-build-push.yml`:

```yaml
env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
  # Add your Docker Hub username if using multi-registry
  DOCKERHUB_NAMESPACE: your-dockerhub-username
```

#### Update Deployment Hosts

Edit `.github/workflows/deploy.yml` and update the server configurations:

```yaml
# Update staging deployment section
- name: Deploy to staging server
  uses: appleboy/ssh-action@v1.0.0
  with:
    host: ${{ secrets.STAGING_HOST }}
    username: ${{ secrets.STAGING_USER }}
    key: ${{ secrets.STAGING_SSH_KEY }}
    port: ${{ secrets.STAGING_PORT || 22 }}
```

#### Configure Notification Channels

Update Slack channel names in all workflows:

```yaml
- name: Notify deployment status
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    channel: '#your-deployment-channel'  # Update this
```

### Step 6: Database Setup

#### Staging Database

```bash
# Connect to staging server
ssh ubuntu@staging.yourdomain.com

# Install PostgreSQL (if using PostgreSQL)
sudo apt install postgresql postgresql-contrib
sudo -u postgres createuser --interactive mlops_user
sudo -u postgres createdb mlops_staging
sudo -u postgres psql -c "ALTER USER mlops_user PASSWORD 'secure_staging_password';"
```

#### Production Database

```bash
# Connect to production server
ssh ubuntu@production.yourdomain.com

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib
sudo -u postgres createuser --interactive mlops_user
sudo -u postgres createdb mlops_production
sudo -u postgres psql -c "ALTER USER mlops_user PASSWORD 'secure_production_password';"
```

## 🧪 Testing the Pipeline

### Step 7: Initial Pipeline Test

#### Test CI Workflow

1. **Create a Test Branch**
   ```bash
   git checkout -b test-ci-pipeline
   echo "# Test CI Pipeline" >> TEST_CI.md
   git add TEST_CI.md
   git commit -m "test: trigger CI pipeline"
   git push origin test-ci-pipeline
   ```

2. **Create Pull Request**
   - Go to GitHub and create a PR from `test-ci-pipeline` to `main`
   - Watch the PR checks workflow execute
   - Verify all checks pass

3. **Monitor Workflow Execution**
   - Go to **Actions** tab in your repository
   - Click on the running workflow
   - Monitor each job's progress and logs

#### Test Docker Build

1. **Trigger Docker Build**
   ```bash
   git checkout main
   git merge test-ci-pipeline
   git push origin main
   ```

2. **Verify Images**
   - Check GitHub Container Registry: `https://github.com/your-username/your-repo/pkgs/container/your-repo`
   - Verify both GPU and CPU variants are built

#### Test Deployment

1. **Deploy to Staging**
   - Push to main branch should automatically trigger staging deployment
   - Monitor deployment logs in Actions tab

2. **Verify Staging Deployment**
   ```bash
   # Test staging endpoints
   curl https://staging.yourdomain.com/health
   curl https://staging.yourdomain.com/model/info
   ```

3. **Test Production Deployment**
   ```bash
   # Create a release tag
   git tag v1.0.0
   git push origin v1.0.0
   ```
   - Go to Actions tab and approve production deployment
   - Monitor production deployment

### Step 8: Performance Testing

#### Run Load Tests

```bash
# Install locust locally
pip install locust

# Run performance test against staging
locust -f tests/locustfile.py --headless -u 50 -r 5 -t 60s --host https://staging.yourdomain.com
```

#### Monitor Performance

- Check Grafana dashboard: `https://staging.yourdomain.com:3000`
- Monitor Prometheus metrics: `https://staging.yourdomain.com:9090`

## 📊 Monitoring and Maintenance

### Step 9: Setup Monitoring

#### Grafana Dashboard Setup

1. **Access Grafana**
   - URL: `https://your-domain.com:3000`
   - Username: `admin`
   - Password: From `GRAFANA_ADMIN_PASSWORD` secret

2. **Import Dashboards**
   - Import pre-built dashboards for:
     - API performance metrics
     - GPU utilization
     - Container health
     - Database performance

#### Prometheus Configuration

Create `docker/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mlops-api'
    static_configs:
      - targets: ['mlops-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['nvidia-gpu-exporter:9445']
    scrape_interval: 10s
```

### Step 10: Automated Maintenance

#### Dependency Updates

The pipeline automatically:
- Checks for dependency updates daily at 2 AM UTC
- Creates PRs for security updates
- Monitors license compliance
- Scans for vulnerabilities

#### Security Monitoring

- **Daily Security Scans**: Automated vulnerability scanning
- **Secret Scanning**: TruffleHog integration
- **License Compliance**: Automated license checking
- **SBOM Generation**: Software Bill of Materials for releases

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Workflow Fails with Permission Errors

**Problem**: `Error: Resource not accessible by integration`

**Solution**:
```bash
# Check repository permissions
# Go to Settings → Actions → General
# Ensure "Read and write permissions" is selected for GITHUB_TOKEN
```

#### 2. Docker Build Fails

**Problem**: `Error: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully`

**Solution**:
```bash
# Check requirements.txt for version conflicts
# Test locally:
docker build -f Dockerfile.cpu -t test-build .
```

#### 3. SSH Connection Fails

**Problem**: `ssh: connect to host staging.yourdomain.com port 22: Connection refused`

**Solution**:
```bash
# Verify SSH key format in GitHub secrets
# Test SSH connection manually:
ssh -i ~/.ssh/id_rsa ubuntu@staging.yourdomain.com

# Check server SSH configuration:
sudo systemctl status ssh
sudo ufw status
```

#### 4. GPU Tests Fail

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Check GPU memory usage
nvidia-smi

# Restart Docker with GPU support
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

#### 5. Database Connection Fails

**Problem**: `psycopg2.OperationalError: could not connect to server`

**Solution**:
```bash
# Check database service
sudo systemctl status postgresql

# Verify database credentials
sudo -u postgres psql -c "\l"

# Check network connectivity
telnet your-db-host 5432
```

### Debug Commands

```bash
# View workflow logs
gh run list --workflow=ci.yml
gh run view [RUN_ID] --log

# Download artifacts
gh run download [RUN_ID]

# Re-run failed jobs
gh run rerun [RUN_ID] --failed

# Check workflow syntax
python scripts/validate-workflows.py

# Test Docker builds locally
docker build -f Dockerfile -t test-gpu .
docker build -f Dockerfile.cpu -t test-cpu .

# Test API locally
python -m uvicorn src.api.main:app --reload
curl http://localhost:8000/health
```

## 📋 Best Practices

### Security Best Practices

1. **Secret Management**
   - Use GitHub secrets for all sensitive data
   - Rotate secrets regularly
   - Use environment-specific secrets

2. **Access Control**
   - Enable branch protection rules
   - Require PR reviews for production deployments
   - Use least-privilege principle for service accounts

3. **Vulnerability Management**
   - Enable automated security scanning
   - Monitor dependency vulnerabilities
   - Keep base images updated

### Performance Best Practices

1. **Workflow Optimization**
   - Use caching for dependencies
   - Parallelize independent jobs
   - Optimize Docker layer caching

2. **Resource Management**
   - Set appropriate resource limits
   - Monitor workflow execution times
   - Use self-hosted runners for heavy workloads

3. **Monitoring**
   - Set up comprehensive monitoring
   - Create alerting rules
   - Monitor key performance indicators

### Maintenance Best Practices

1. **Regular Updates**
   - Update GitHub Actions versions monthly
   - Review and update dependencies weekly
   - Monitor security advisories

2. **Documentation**
   - Keep documentation up to date
   - Document any customizations
   - Maintain runbooks for common issues

3. **Testing**
   - Test pipeline changes in feature branches
   - Maintain comprehensive test coverage
   - Regular disaster recovery testing

## 📚 Additional Resources

### Documentation Links

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [Deployment Environments](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment)
- [Security Hardening](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)

### Monitoring Tools

- [Grafana Dashboards](https://grafana.com/grafana/dashboards/)
- [Prometheus Configuration](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)
- [NVIDIA GPU Monitoring](https://github.com/mindprince/nvidia_gpu_prometheus_exporter)

### Troubleshooting Resources

- [GitHub Actions Troubleshooting](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/troubleshooting-workflows)
- [Docker Troubleshooting](https://docs.docker.com/config/troubleshooting/)
- [SSH Troubleshooting](https://docs.github.com/en/authentication/troubleshooting-ssh)

---

## 🎉 Conclusion

This comprehensive CI/CD pipeline provides enterprise-grade automation for the MLOps California Housing Platform. With proper setup and configuration, it enables:

- **Automated Testing**: Comprehensive code quality and security checks
- **Reliable Deployments**: Consistent and safe deployments to multiple environments
- **Performance Monitoring**: Continuous performance tracking and optimization
- **Security Compliance**: Automated vulnerability scanning and compliance checking

Follow this guide step by step to set up a robust CI/CD pipeline that will scale with your MLOps platform needs.

For additional support or questions, refer to the troubleshooting section or create an issue in the repository.