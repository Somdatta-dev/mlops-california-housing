# VS Code Conda Environment Setup

## 🎯 To use your mlops conda environment in VS Code:

### Method 1: Select Python Interpreter
1. Open VS Code in your project folder
2. Press `Ctrl+Shift+P` to open Command Palette
3. Type "Python: Select Interpreter"
4. Choose: `C:\Users\Sumit\anaconda3\envs\mlops\python.exe`

### Method 2: Create .vscode/settings.json
Create a `.vscode/settings.json` file in your project root with:

```json
{
    "python.defaultInterpreterPath": "C:\\Users\\Sumit\\anaconda3\\envs\\mlops\\python.exe",
    "python.terminal.activateEnvironment": true
}
```

### Method 3: Use Terminal in VS Code
1. Open terminal in VS Code (`Ctrl+``)
2. Run: `C:\Users\Sumit\anaconda3\Scripts\conda.exe activate mlops`
3. Then run your Python scripts

## 🧪 Test Your Setup

Run this to verify mlops environment is working:
```bash
C:\Users\Sumit\anaconda3\envs\mlops\python.exe -c "import xgboost, mlflow, torch; print('✅ All packages available!')"
```

## 🚀 Run XGBoost Example with mlops Environment

```bash
C:\Users\Sumit\anaconda3\envs\mlops\python.exe examples/xgboost_gpu_example.py
```

## 📋 Available Environments
- **mlops** ← Use this one (has XGBoost, MLflow, PyTorch)
- dev2 (currently active in terminal)
- dev3
- base
- jupyter
- autogen
- crewai