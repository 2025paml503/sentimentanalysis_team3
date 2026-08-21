# Installation Guide

This project uses a Python virtual environment to keep dependencies isolated.

## 1) Create a virtual environment

From the project root:

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

- Windows Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

- macOS/Linux:
```bash
source .venv/bin/activate
```

## 2) Upgrade pip

```bash
python -m pip install --upgrade pip
```

## 3) Install project dependencies

From the project root:

```bash
pip install -r requirements.txt
```

If you also want to install the project in editable mode later, you can use:

```bash
pip install -e .
```

## 4) Verify installation

```bash
python -c "import pandas, sklearn, dvc, mlflow; print('Dependencies installed successfully')"
```

## 5) Deactivate the environment

```bash
deactivate
```
