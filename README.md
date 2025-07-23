# MLOps California Housing Prediction Platform

A complete MLOps pipeline for California Housing price prediction with DVC data versioning, MLflow experiment tracking, FastAPI deployment, and comprehensive monitoring.

## 🏗️ Architecture Overview

```
MLOps Platform
├── Data Management (DVC)
│   ├── California Housing Dataset
│   ├── Data Validation & Quality Checks
│   └── Version Control with Remote Storage
├── Model Development (MLflow)
│   ├── Multiple ML Models (XGBoost, Neural Networks, etc.)
│   ├── Experiment Tracking
│   └── Model Registry
├── API Deployment (FastAPI)
│   ├── Prediction Endpoints
│   ├── Model Serving
│   └── Input Validation
├── CI/CD Pipeline (GitHub Actions)
│   ├── Automated Testing
│   ├── Docker Build & Deploy
│   └── Quality Gates
└── Monitoring & Logging
    ├── Prediction Logging
    ├── Model Performance Tracking
    └── System Health Monitoring
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- 4GB+ RAM (for ML model training)

### 1. Clone Repository
```bash
git clone https://github.com/Somdatta-dev/mlops-california-housing.git
cd mlops-california-housing
```

### 2. Automated Setup (Recommended)
```bash
python setup_project.py
```

This script will:
- ✅ Install all dependencies
- ✅ Set up DVC remote storage
- ✅ Pull dataset from DVC
- ✅ Verify data integrity
- ✅ Test data loading

### 3. Manual Setup (Alternative)
```bash
# Install dependencies
pip install -r requirements.txt

# Create DVC remote storage
mkdir ../dvc_remote_storage

# Pull data from DVC
dvc pull

# Verify setup
python src/data_validation.py
```

## 📊 Dataset Information

**California Housing Dataset**
- **Samples**: 20,640 housing records
- **Features**: 8 numerical features
- **Target**: Median house value
- **Source**: sklearn.datasets.fetch_california_housing

**Features:**
- `MedInc`: Median income in block group
- `HouseAge`: Median house age in block group  
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude

## 🔧 Project Structure

```
mlops-california-housing/
├── data/
│   ├── raw/                    # Raw dataset files (DVC tracked)
│   └── processed/              # Processed data (DVC tracked)
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Data loading utilities
│   │   └── data_validation.py  # Data quality validation
│   ├── models/                 # ML model training scripts
│   ├── api/                    # FastAPI application
│   └── monitoring/             # Logging and monitoring
├── notebooks/                  # Jupyter notebooks for EDA
├── tests/                      # Unit and integration tests
├── docker/                     # Docker configuration
├── .github/workflows/          # CI/CD pipelines
├── requirements.txt            # Python dependencies
├── setup_project.py           # Automated setup script
└── README.md                  # This file
```

## 🛠️ Development Workflow

### Data Management
```bash
# Load and validate data
python src/data_loader.py
python src/data_validation.py

# Check DVC status
dvc status
dvc pull  # Pull latest data
dvc push  # Push data changes
```

### Model Training
```bash
# Start MLflow tracking server
mlflow ui --host 0.0.0.0 --port 5000

# Train models (when implemented)
python src/models/train_models.py
```

### API Development
```bash
# Start FastAPI server (when implemented)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run tests (when implemented)
pytest tests/
pytest --cov=src tests/  # With coverage
```

## 📈 Data Validation Results

The dataset passes comprehensive validation with the following checks:
- ✅ **Schema Validation**: All expected columns present with correct types
- ✅ **Data Quality**: No missing values, no duplicates, no infinite values
- ✅ **Statistical Properties**: Reasonable distributions and correlations
- ⚠️ **Value Ranges**: Some outliers detected (normal for real-world data)

**Dataset Statistics:**
- Total samples: 20,640
- Features: 8 numerical
- Missing values: 0
- Duplicate rows: 0
- Data quality score: 100%

## 🔄 DVC Data Versioning

This project uses DVC (Data Version Control) for dataset management:

**Current Configuration:**
- **Remote Storage**: Local directory (`../dvc_remote_storage`)
- **Tracked Files**: 
  - `california_housing_features.csv` (1.8MB)
  - `california_housing_targets.csv` (164KB)
  - `dataset_metadata.json` (2KB)
  - `validation_report.json` (4KB)

**DVC Commands:**
```bash
dvc status    # Check data status
dvc pull      # Download data from remote
dvc push      # Upload data to remote
dvc add       # Track new data files
```

## 🐳 Docker Support

```bash
# Build Docker image (when implemented)
docker build -t mlops-housing .

# Run container
docker run -p 8000:8000 mlops-housing
```

## 🔍 Monitoring & Logging

The platform includes comprehensive monitoring:
- **Data Quality**: Automated validation reports
- **Model Performance**: MLflow experiment tracking
- **API Metrics**: Request/response logging
- **System Health**: Resource usage monitoring

## 🧪 Testing Strategy

```bash
# Unit tests
pytest tests/unit/

# Integration tests  
pytest tests/integration/

# Data validation tests
pytest tests/data/

# API tests (when implemented)
pytest tests/api/
```

## 📚 API Documentation

Once the API is implemented, documentation will be available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🚀 Deployment

### Local Development
```bash
python setup_project.py
uvicorn src.api.main:app --reload
```

### Production (Docker)
```bash
docker-compose up -d
```

### Cloud Deployment
- AWS EC2 with Docker
- Google Cloud Run
- Azure Container Instances

## 🔧 Configuration

Environment variables (`.env`):
```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=california-housing-prediction

# API Configuration  
API_HOST=0.0.0.0
API_PORT=8000

# Database Configuration
DATABASE_URL=sqlite:///./mlops_platform.db
```



## 🆘 Troubleshooting

### Common Issues

**DVC Pull Fails:**
```bash
# Check remote configuration
dvc remote list

# Recreate remote storage
mkdir ../dvc_remote_storage
dvc push
```

**Data Loading Errors:**
```bash
# Verify data files exist
ls -la data/raw/

# Re-download dataset
python src/data_loader.py
```

**Import Errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```


Built with ❤️ for MLOps best practices