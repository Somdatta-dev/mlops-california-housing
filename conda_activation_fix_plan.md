# VSCode Conda Activation Fix Plan

## Problem Summary
VSCode is automatically trying to activate conda environment when opening new terminals, causing failures because:
- `"python.terminal.activateEnvironment": true` forces automatic activation
- `"python.terminal.activateEnvInCurrentTerminal": true` activates in current terminals
- `conda` command is not available in system PATH
- Results in error: `conda: The term 'conda' is not recognized...`

## Current Problematic Settings
Located in `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "C:\\Users\\Sumit\\anaconda3\\envs\\mlops\\python.exe",
    "python.terminal.activateEnvironment": true,           // ← REMOVE THIS
    "python.terminal.activateEnvInCurrentTerminal": true,  // ← REMOVE THIS
    "python.condaPath": "C:\\Users\\Sumit\\anaconda3\\Scripts\\conda.exe", // ← REMOVE THIS
    "python.envFile": "${workspaceFolder}/.env",
    // ... other settings ...
}
```

## Solution: Modified Settings
Keep essential settings, remove automatic activation:
```json
{
    "python.defaultInterpreterPath": "C:\\Users\\Sumit\\anaconda3\\envs\\mlops\\python.exe",
    "python.envFile": "${workspaceFolder}/.env",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "files.encoding": "utf8",
    "terminal.integrated.env.windows": {
        "PYTHONPATH": "${workspaceFolder}"
    }
}
```

## Implementation Steps
1. ✅ **Backup current settings** - Copy `.vscode/settings.json` to `.vscode/settings.json.backup`
2. ✅ **Remove problematic settings**:
   - `"python.terminal.activateEnvironment": true`
   - `"python.terminal.activateEnvInCurrentTerminal": true` 
   - `"python.condaPath": "C:\\Users\\Sumit\\anaconda3\\Scripts\\conda.exe"`
3. ✅ **Keep beneficial settings**:
   - `"python.defaultInterpreterPath"` - for code editing/debugging
   - All testing, linting, and formatting settings
4. ✅ **Test the fix** - Open new terminal and verify no conda activation attempts

## Manual Activation Options (When Needed)
If you need to activate conda manually later:

### Option 1: Direct Python Path
```bash
C:\Users\Sumit\anaconda3\envs\mlops\python.exe your_script.py
```

### Option 2: Manual Conda Activation
```bash
C:\Users\Sumit\anaconda3\Scripts\activate.bat mlops
```

### Option 3: Add Conda to PATH (Global Fix)
Add to system PATH: `C:\Users\Sumit\anaconda3\Scripts`

## Expected Outcome
- ✅ New terminals open without conda activation attempts
- ✅ No more "conda: The term 'conda' is not recognized" errors
- ✅ VSCode Python features still work (debugging, linting, testing)
- ✅ Manual activation available when needed

## Rollback Plan
If issues arise, restore from backup:
```bash
copy .vscode\settings.json.backup .vscode\settings.json