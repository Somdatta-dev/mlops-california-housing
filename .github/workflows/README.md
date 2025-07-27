# GitHub Actions CI/CD Pipeline

This directory contains the comprehensive CI/CD pipeline for the MLOps California Housing Platform. The pipeline implements automated testing, building, deployment, and monitoring workflows.

## 📋 Workflow Overview

### Core Workflows

| Workflow | Trigger | Purpose | Duration |
|----------|---------|---------|----------|
| [`ci.yml`](./ci.yml) | Push, PR | Code quality, testing, security | ~15-20 min |
| [`docker-build-push.yml`](./docker-build-push.yml) | Push, tags | Build and push Docker images | ~10-15 min |
| [`deploy.yml`](./deploy.yml) | Main branch, tags | Deploy to staging/production | ~5-10 min |
| [`pr-checks.yml`](./pr-checks.yml) | Pull requests | PR validation and testing | ~10-15 min |
| [`release.yml`](./release.yml) | Tags (v*) | Create releases and deploy | ~20-30 min |

### Maintenance Workflows

| Workflow | Trigger | Purpose | Duration |
|----------|---------|---------|----------|
| [`dependency-update.yml`](./dependency-update.yml) | Daily, manual | Security and dependency updates | ~5-10 min |
| [`workflow-dispatch.yml`](./workflow-dispatch.yml) | Manual | On-demand workflow execution | Variable |

## 🚀 Workflow Details

### 1. CI Workflow (`ci.yml`)

**Triggers**: Push to main/develop, Pull Requests

**Jobs**:
- **Code Quality**: Black, Flake8, MyPy, Bandit, Safety
- **Unit Tests**: Multi-Python version testing (3.9, 3.10, 3.11)
- **Integration Tests**: API testing with PostgreSQL
- **Docker Build Test**: Verify containerization
- **Documentation Check**: Validate README and docs
- **Security Scan**: Trivy vulnerability scanning
- **Performance Test**: Basic load testing with Locust

**Artifacts**:
- Test coverage reports
- Security scan results
- Performance benchmarks

### 2. Docker Build and Push (`docker-build-push.yml`)

**Triggers**: Push to main/develop, Tags, Manual

**Features**:
- Multi-architecture builds (AMD64, ARM64 for CPU variant)
- GPU and CPU variants
- Multi-registry support (GitHub Container Registry, Docker Hub)
- Security scanning with Trivy
- Performance benchmarking
- SBOM generation

**Images Built**:
- `ghcr.io/[repo]:latest-gpu` - GPU-enabled production image
- `ghcr.io/[repo]:latest-cpu` - CPU-only production image
- Tagged versions for releases

### 3. Deployment (`deploy.yml`)

**Triggers**: Main branch pushes, Release tags, Manual

**Environments**:
- **Staging**: Automatic deployment from main branch
- **Production**: Manual approval required, triggered by tags

**Features**:
- Rolling deployments with health checks
- Automatic rollback on failure
- Environment-specific configuration
- Smoke testing after deployment
- Slack notifications

**Deployment Process**:
1. Backup current deployment
2. Pull new Docker image
3. Update configuration
4. Rolling deployment
5. Health checks and smoke tests
6. Notification and monitoring

### 4. Pull Request Checks (`pr-checks.yml`)

**Triggers**: PR opened, synchronized, reopened

**Comprehensive Checks**:
- **Change Detection**: Identify modified components
- **Code Quality**: Focused on changed files only
- **Unit Testing**: Full test suite with coverage
- **Docker Build**: Verify containerization works
- **Security Scan**: Vulnerability assessment
- **Performance Impact**: Compare with base branch
- **API Contract Testing**: Validate API compatibility
- **Documentation**: Check for doc updates

**PR Status Updates**:
- Automated comments with check results
- GitHub status checks integration
- Performance comparison reports

### 5. Release Management (`release.yml`)

**Triggers**: Version tags (v*), Manual dispatch

**Release Process**:
1. **Validation**: Version format, changelog, tag availability
2. **Testing**: Full test suite execution
3. **Building**: Multi-platform Docker images
4. **Security**: Comprehensive vulnerability scanning
5. **Release Creation**: GitHub release with artifacts
6. **Deployment**: Automatic production deployment (stable releases)
7. **Documentation**: Update version references
8. **Notification**: Slack and GitHub notifications

**Artifacts**:
- Source code archives with checksums
- Docker images for GPU and CPU variants
- Security scan reports
- SBOM (Software Bill of Materials)

### 6. Dependency Updates (`dependency-update.yml`)

**Triggers**: Daily schedule (2 AM UTC), Manual, Requirements changes

**Features**:
- **Dependency Scanning**: Check for outdated packages
- **Security Monitoring**: Identify vulnerable dependencies
- **Automated PRs**: Create update pull requests
- **License Compliance**: Validate license compatibility
- **Docker Base Images**: Monitor base image updates

**Security Monitoring**:
- Bandit static analysis
- Safety vulnerability database
- Semgrep security patterns
- Secret scanning with TruffleHog
- Automated issue creation for critical findings

## 🔧 Configuration

### Required Secrets

#### GitHub Secrets
```bash
# Docker Registry
DOCKERHUB_USERNAME          # Docker Hub username
DOCKERHUB_TOKEN            # Docker Hub access token

# Deployment
STAGING_HOST               # Staging server hostname
STAGING_USER               # SSH username for staging
STAGING_SSH_KEY           # SSH private key for staging
STAGING_PORT              # SSH port (default: 22)

PROD_HOST                 # Production server hostname
PROD_USER                 # SSH username for production
PROD_SSH_KEY             # SSH private key for production
PROD_PORT                # SSH port (default: 22)

# Database
STAGING_DB_HOST           # Staging database host
STAGING_DB_USER           # Staging database user
STAGING_DB_PASSWORD       # Staging database password
STAGING_DB_NAME           # Staging database name

PROD_DB_HOST             # Production database host
PROD_DB_USER             # Production database user
PROD_DB_PASSWORD         # Production database password
PROD_DB_NAME             # Production database name

# API Keys
STAGING_API_KEY          # API key for staging environment
PROD_API_KEY            # API key for production environment

# Monitoring
GRAFANA_ADMIN_PASSWORD   # Grafana admin password

# DVC
DVC_REMOTE_URL          # Google Drive DVC remote URL

# Notifications
SLACK_WEBHOOK_URL       # Slack webhook for notifications
```

#### Environment Variables
```bash
# Python Configuration
PYTHON_VERSION=3.10

# Registry Configuration
REGISTRY=ghcr.io
IMAGE_NAME=${{ github.repository }}
```

### GitHub Environments

#### Staging Environment
- **Protection Rules**: None (automatic deployment)
- **Secrets**: Staging-specific credentials
- **URL**: https://staging.mlops-california-housing.example.com

#### Production Environment
- **Protection Rules**: Required reviewers, deployment branches
- **Secrets**: Production credentials
- **URL**: https://mlops-california-housing.example.com

## 📊 Monitoring and Observability

### Workflow Monitoring
- **GitHub Actions**: Built-in workflow monitoring
- **Slack Notifications**: Real-time deployment and security alerts
- **Artifact Storage**: Test reports, security scans, performance data

### Deployment Monitoring
- **Health Checks**: Automated endpoint validation
- **Smoke Tests**: Post-deployment functionality verification
- **Performance Monitoring**: Load testing and benchmarking
- **Security Scanning**: Continuous vulnerability assessment

## 🛠️ Usage Examples

### Manual Workflow Dispatch
```bash
# Trigger CI workflow manually
gh workflow run ci.yml

# Deploy to staging with specific image
gh workflow run deploy.yml -f environment=staging -f image_tag=v1.2.3

# Run security scan
gh workflow run workflow-dispatch.yml -f workflow_type=security-scan

# Performance testing
gh workflow run workflow-dispatch.yml -f workflow_type=performance-test -f test_type=performance
```

### Release Process
```bash
# Create and push a release tag
git tag v1.0.0
git push origin v1.0.0

# This triggers:
# 1. Release workflow
# 2. Docker image building
# 3. Security scanning
# 4. GitHub release creation
# 5. Production deployment (if stable)
```

### Emergency Procedures

#### Rollback Production
```bash
# Manual rollback via workflow
gh workflow run deploy.yml -f environment=production -f image_tag=v1.0.0

# Or SSH to production server
ssh prod-server
cd /opt/mlops-production
# Restore from backup (automated in workflow)
```

#### Security Incident Response
1. **Immediate**: Disable affected workflows
2. **Assessment**: Review security scan reports
3. **Mitigation**: Apply security patches via dependency updates
4. **Validation**: Run comprehensive security scans
5. **Deployment**: Emergency deployment with security fixes

## 🔍 Troubleshooting

### Common Issues

#### Docker Build Failures
- Check Dockerfile syntax
- Verify base image availability
- Review dependency conflicts
- Check disk space on runners

#### Deployment Failures
- Verify server connectivity
- Check SSH key permissions
- Validate environment variables
- Review server logs

#### Test Failures
- Check test environment setup
- Verify database connectivity
- Review dependency versions
- Check GPU availability (for GPU tests)

### Debug Commands
```bash
# Check workflow status
gh run list --workflow=ci.yml

# View workflow logs
gh run view [RUN_ID] --log

# Download artifacts
gh run download [RUN_ID]

# Re-run failed jobs
gh run rerun [RUN_ID] --failed
```

## 📈 Performance Metrics

### Workflow Performance Targets
- **CI Pipeline**: < 20 minutes
- **Docker Build**: < 15 minutes
- **Deployment**: < 10 minutes
- **Security Scan**: < 5 minutes

### Success Rate Targets
- **CI Success Rate**: > 95%
- **Deployment Success Rate**: > 99%
- **Security Scan Coverage**: 100%

## 🔄 Maintenance

### Regular Tasks
- **Weekly**: Review failed workflows and performance metrics
- **Monthly**: Update workflow dependencies and actions versions
- **Quarterly**: Review and update security policies
- **Annually**: Comprehensive pipeline architecture review

### Workflow Updates
1. Test changes in feature branches
2. Use workflow dispatch for validation
3. Monitor performance impact
4. Update documentation
5. Communicate changes to team

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [Deployment Best Practices](https://docs.github.com/en/actions/deployment/about-deployments)
- [Security Hardening](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)

---

**Note**: This CI/CD pipeline is designed for the MLOps California Housing Platform and implements enterprise-grade practices for automated testing, building, and deployment. Regular maintenance and monitoring ensure optimal performance and security.