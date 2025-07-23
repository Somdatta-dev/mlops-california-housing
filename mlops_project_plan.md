# Complete MLOps Project Plan - California Housing Prediction
*Total Possible Score: 30 marks (26 base + 4 bonus)*

## 🎯 Project Overview
**Dataset**: California Housing (Regression)  
**Objective**: Build a complete MLOps pipeline for house price prediction with production-ready deployment, monitoring, and automated retraining.

---

## 📋 Part 1: Repository and Data Versioning (4 marks)

### Tasks:
- [ ] **GitHub Repository Setup**
  - Create repo: `mlops-california-housing`
  - Initialize with README, .gitignore (Python template)
  - Set up branch protection rules

- [ ] **Dataset Management**
  - Download California Housing dataset from sklearn
  - Store in `data/raw/` directory
  - Create data loading script with version tracking

- [ ] **DVC Integration** 
  - Install DVC: `pip install dvc`
  - Initialize DVC: `dvc init`
  - Track dataset: `dvc add data/raw/housing.csv`
  - Configure remote storage (DVC with Google Drive/S3)
  - Commit .dvc files to Git

- [ ] **Directory Structure**
  ```
  mlops-california-housing/
  ├── data/
  │   ├── raw/
  │   └── processed/
  ├── src/
  │   ├── data/
  │   ├── models/
  │   ├── api/
  │   └── monitoring/
  ├── notebooks/
  ├── tests/
  ├── docker/
  ├── .github/workflows/
  ├── requirements.txt
  ├── Dockerfile
  └── README.md
  ```

---

## 🤖 Part 2: Model Development & Experiment Tracking (6 marks)

### Models to Train:
1. **Linear Regression** (GPU-accelerated with cuML for large datasets)
2. **Random Forest Regressor** (GPU-accelerated with cuML for high n_estimators)
3. **XGBoost Regressor** (GPU-accelerated with tree_method='gpu_hist')
4. **Neural Network Regressor** (PyTorch GPU - deep architecture)
5. **LightGBM** (GPU-accelerated for comparison)

### High-Performance Training Configuration:
- [ ] **GPU Optimization Setup**
  ```python
  # XGBoost GPU Configuration
  xgb_params = {
      'tree_method': 'gpu_hist',
      'gpu_id': 0,
      'max_depth': 12,           # Deeper trees with 32GB VRAM
      'n_estimators': 5000,      # More trees for better accuracy
      'learning_rate': 0.01,     # Lower LR, more iterations
      'subsample': 0.8,
      'colsample_bytree': 0.8,
      'objective': 'reg:squarederror'
  }
  
  # Neural Network Configuration
  nn_config = {
      'hidden_layers': [512, 256, 128, 64],  # Deep architecture
      'batch_size': 2048,        # Large batches with 32GB VRAM  
      'epochs': 500,             # Extended training
      'early_stopping': True,
      'device': 'cuda',
      'precision': 'mixed'       # Mixed precision for efficiency
  }
  ```

- [ ] **Hardware Utilization Monitoring**
  - GPU memory usage tracking
  - CUDA utilization metrics
  - Temperature and power monitoring
  - Training throughput (samples/second)

### MLflow Setup:
- [ ] **MLflow Installation & Configuration**
  - Install: `pip install mlflow`
  - Set up tracking server: `mlflow server --host 0.0.0.0 --port 5000`
  - Configure experiment: "California Housing Prediction"

- [ ] **Experiment Tracking Script** (`src/models/train_models.py`)
  - Log hyperparameters for all models
  - Track metrics: RMSE, MAE, R²
  - Log model artifacts and GPU utilization
  - Save feature importance plots and training curves
  - Record training time, GPU memory usage, and data version
  - Enable mixed precision training for neural networks
  - Implement gradient accumulation for large effective batch sizes

- [ ] **Model Registration**
  - Register best performing model in MLflow Model Registry
  - Tag with version and performance metrics
  - Set model stage to "Staging"

### Key Metrics to Track:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error) 
- R² Score
- Training/Validation split performance
- **GPU Metrics**: Memory usage, utilization %, training speed
- **Hardware Monitoring**: Temperature, power consumption
- **Training Efficiency**: Samples/second, time per epoch

---

## 🚀 Part 3: API & Docker Packaging (4 marks)

### FastAPI Development:
- [ ] **Create API** (`src/api/main.py`)
  ```python
  # Endpoints:
  # POST /predict - Single prediction
  # POST /predict/batch - Batch predictions
  # GET /health - Health check
  # GET /model/info - Model metadata
  ```

- [ ] **Input Validation with Pydantic**
  - Create data models for input validation
  - Handle edge cases and error responses
  - Add request/response examples

- [ ] **Docker Configuration**
  - Create optimized Dockerfile with CUDA support
  - Multi-stage build for smaller image
  - Include MLflow model loading and GPU libraries
  - Set up Docker Compose with GPU passthrough
  - NVIDIA Container Runtime configuration

- [ ] **API Testing**
  - Unit tests for endpoints
  - Integration tests with model predictions
  - Load testing with sample data

---

## 🔄 Part 4: CI/CD with GitHub Actions (6 marks)

### GitHub Actions Workflows:

- [ ] **Linting & Testing** (`.github/workflows/test.yml`)
  ```yaml
  # Triggers: Push, PR to main
  # Jobs: Lint code, run tests, check coverage
  ```

- [ ] **Docker Build & Push** (`.github/workflows/docker.yml`)
  ```yaml
  # Build Docker image
  # Push to Docker Hub/GitHub Container Registry
  # Tag with commit SHA and latest
  ```

- [ ] **Deployment Pipeline** (`.github/workflows/deploy.yml`)
  ```yaml
  # Deploy to staging environment
  # Run integration tests
  # Deploy to production (manual approval)
  ```

### Deployment Options:
- **Local**: Docker Compose
- **Cloud**: Deploy to AWS EC2/Google Cloud Run/Azure Container Instances

---

## 📊 Part 5: Logging and Monitoring (4 marks)

### Logging Implementation:
- [ ] **Structured Logging** (`src/monitoring/logger.py`)
  - Request/response logging
  - Model prediction logging
  - Error tracking and alerting
  - Performance metrics logging

- [ ] **Database Setup**
  - SQLite for development
  - Log prediction requests with timestamps
  - Store model performance metrics

- [ ] **Optional: Metrics Endpoint**
  - Expose Prometheus metrics
  - Track prediction latency, throughput
  - Model accuracy over time

### Advanced Web Dashboard (Next.js):
- **Real-time predictions display** with charts and tables
- **Interactive training interface** with pause/resume capabilities
- **Database browser** for exploring prediction history
- **Model comparison dashboard** with performance metrics
- **System health monitoring** with live status indicators

---

## 📑 Part 6: Summary + Demo (2 marks)

- [ ] **Documentation** (`README.md`)
  - Project architecture overview
  - Setup and installation instructions
  - API usage examples
  - MLOps pipeline explanation

- [ ] **5-Minute Video Demo**
  - Architecture walkthrough
  - Live API demonstration
  - MLflow experiment tracking
  - CI/CD pipeline overview
  - Monitoring capabilities

---

## ⭐ BONUS FEATURES (4 marks)

### 1. Input Validation with Pydantic/Schema (1 mark)
- [ ] **Advanced Pydantic Models**
  ```python
  # Detailed input validation
  # Custom validators for housing data
  # Comprehensive error messages
  ```

### 2. Prometheus Integration & Dashboard (2 marks)
- [ ] **Prometheus Metrics**
  - Install: `pip install prometheus_client`
  - Custom metrics: prediction_duration, requests_total
  - Model performance metrics

- [ ] **Grafana Dashboard**
  - Docker Compose with Prometheus + Grafana
  - Real-time monitoring dashboard
  - Alerts for model performance degradation

## 🌐 ENHANCED BONUS: Next.js Web Dashboard (Additional 2-3 marks potential)

### Dashboard Features:
- [ ] **Real-time Predictions Dashboard**
  ```typescript
  // Components:
  // - Live prediction feed with charts
  // - Model performance metrics
  // - Prediction accuracy over time
  // - Interactive data visualization
  ```

- [ ] **Interactive Training Interface**
  - Start/Stop/Pause training jobs with GPU monitoring
  - Real-time training progress with loss curves and GPU usage
  - Hyperparameter tuning interface with GPU-optimized presets
  - Model comparison side-by-side with performance metrics
  - GPU memory allocation and utilization graphs

- [ ] **Database Explorer**
  - Browse prediction history with filters
  - Export data to CSV/JSON
  - Search and pagination
  - Connection to multiple data sources (SQLite, PostgreSQL, Cloud)

- [ ] **System Monitoring**
  - Live API health status
  - Resource usage (CPU, Memory, **GPU Memory, GPU Utilization**)
  - Model serving statistics
  - Error logs and alerts
  - **NVIDIA GPU metrics** (temperature, power, compute utilization)

### Technical Implementation:
- [ ] **Next.js 15 Setup** (`dashboard/`)
  ```
  dashboard/
  ├── app/
  │   ├── dashboard/
  │   │   ├── page.tsx (Server Component)
  │   │   └── loading.tsx
  │   ├── training/
  │   │   ├── page.tsx
  │   │   └── components/
  │   ├── database/
  │   │   └── page.tsx
  │   ├── api/
  │   │   ├── predictions/route.ts
  │   │   └── training/route.ts
  │   ├── globals.css
  │   └── layout.tsx
  ├── components/
  │   ├── ui/ (shadcn/ui components)
  │   ├── PredictionDashboard.tsx
  │   ├── TrainingInterface.tsx
  │   ├── DatabaseBrowser.tsx
  │   └── SystemMonitor.tsx
  ├── lib/
  │   ├── api-client.ts
  │   ├── websocket.ts
  │   └── utils.ts
  ├── hooks/
  │   ├── use-websocket.ts
  │   └── use-training-status.ts
  └── package.json
  ```

- [ ] **WebSocket Integration**
  - Real-time updates for training progress
  - Live prediction streaming
  - System health notifications

- [ ] **Backend API Extensions**
  ```python
  # Additional FastAPI endpoints:
  # GET /api/predictions/recent
  # POST /api/training/start
  # POST /api/training/pause
  # GET /api/training/status
  # GET /api/database/browse
  ```

---

## 🗓️ Timeline & Milestones

### Week 1 (July 22-28)
- [ ] Repository setup and DVC integration
- [ ] Data preprocessing and EDA
- [ ] Model development and MLflow setup

### Week 2 (July 29 - Aug 4)
- [ ] API development and Docker packaging
- [ ] CI/CD pipeline setup
- [ ] Next.js dashboard foundation
- [ ] WebSocket integration for real-time updates

### Week 3 (Aug 5-11)
- [ ] Advanced dashboard features (training interface, database browser)
- [ ] Bonus features implementation
- [ ] Testing and refinement
- [ ] Documentation and video recording

---

## 📦 Key Technologies Stack

**Core Technologies:**
- Python 3.9+
- scikit-learn, XGBoost (GPU), LightGBM (GPU)
- **PyTorch (CUDA 12.8 or later)**
- **NVIDIA RAPIDS (cuDF, cuML) for GPU acceleration**
- MLflow
- FastAPI
- Docker with NVIDIA Container Runtime
- GitHub Actions

**Data Management:**
- DVC (Data Version Control)
- SQLite/PostgreSQL
- Redis (for caching and session management)

**Frontend Dashboard:**
- Next.js 15.4 with TypeScript (App Router)
- React 19.0 with latest features
- TailwindCSS + shadcn/ui components
- Chart.js/Recharts for visualizations
- WebSocket for real-time updates
- React 19 features (Server Components, Actions)

**Monitoring (Bonus):**
- Prometheus + **nvidia-ml-py** for GPU metrics
- Grafana with GPU dashboards
- Pydantic validation
- **GPUstat** for real-time monitoring

**Testing:**
- pytest
- coverage
- pre-commit hooks

---

## 🎯 Success Criteria

**Minimum Requirements (26 marks):**
- ✅ Working GitHub repo with DVC
- ✅ 3 trained models tracked in MLflow
- ✅ Dockerized FastAPI with predictions
- ✅ Complete CI/CD pipeline
- ✅ Logging and basic monitoring
- ✅ Documentation and demo video

**Bonus Achievement (30 marks):**
- ✅ Advanced input validation
- ✅ Prometheus/Grafana monitoring
- ✅ Automated retraining pipeline

---

## 💡 Pro Tips for Maximum Marks

1. **Over-deliver on documentation** - Clear README with setup instructions
2. **Make it production-ready** - Error handling, logging, validation
3. **Show MLOps best practices** - Model versioning, experiment tracking
4. **Demonstrate monitoring** - Real metrics and dashboards
5. **Clean, professional code** - Type hints, docstrings, tests

This comprehensive plan ensures you hit every requirement while positioning you for the full 30 marks!
