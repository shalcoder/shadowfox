# ChestMNIST AI Classifier - Phase 1 Setup

## Project Overview
This project is an AI-powered diagnostic platform that analyzes chest X-ray images to detect 14 different pathologies using deep learning. It uses a TensorFlow model and a Streamlit frontend.

## Prerequisites
- **Python 3.8** or higher installed on your system.
- **Git** (optional, for cloning).

## Setup Instructions

Follow these steps to set up the project from scratch.

### 1. Navigate to the Project Directory
Open your terminal (Command Prompt or PowerShell) and navigate to the project root:
```cmd
cd e:\shadowfox\phase1\image_tagger
```
*(Note: Adjust the path if you have placed the project elsewhere)*

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

**Windows (cmd/PowerShell):**
```cmd
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate
```

### 4. Install Dependencies
Install the required Python packages using the `requirements.txt` file located in the `frontend` folder.

```cmd
pip install -r frontend/requirements.txt
```

---

## Running the Application

**IMPORTANT:** You must run the application from the **root directory** (`image_tagger`) for the model paths to work correctly.

Run the following command:

```cmd
streamlit run frontend/app.py
```

The application should automatically open in your default web browser at `http://localhost:8501`.

---

## Troubleshooting

### 1. "Model not found" or Path Errors
**Error:** `Error loading model: ...` or `Failed to load the model.`
**Solution:**
- Ensure you are running the `streamlit run` command from the **root directory** (`image_tagger`), NOT from inside the `frontend` folder.
- Verify that the model files exist at `frontend/model/saved_model/v2`.

### 2. "streamlit is not recognized"
**Error:** `'streamlit' is not recognized as an internal or external command...`
**Solution:**
- Ensure your virtual environment is **activated**. You should see `(venv)` at the start of your command line.
- Try running with python module syntax: `python -m streamlit run frontend/app.py`.

### 3. Dependency Conflicts
**Error:** Errors during `pip install`.
**Solution:**
- Upgrade pip: `python -m pip install --upgrade pip`
- Try installing dependencies individually if the bulk install fails.

### 4. Port already in use
**Error:** `Port 8501 is already in use`
**Solution:**
- Streamlit will automatically try the next available port (e.g., 8502). Check the terminal output for the correct URL.
- You can specify a different port manually:
  ```cmd
  streamlit run frontend/app.py --server.port 8505
  ```

## Project Structure
```
image_tagger/
├── frontend/
│   ├── app.py              # Main Streamlit application
│   ├── requirements.txt    # Python dependencies
│   └── model/              # ML Model files
├── notebooks/              # Jupyter notebooks for research
└── README.md               # This file
```
