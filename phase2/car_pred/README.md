# Shadowfox - Car Price Prediction (Phase 2)

This project is a complete pipeline for predicting second-hand car prices. It consists of a Machine Learning model, a FastAPI backend, and a Streamlit frontend user interface.

## Prerequisites
- **Python 3.8** or higher
- **Git** (optional, for cloning)

## Setup & Installation

Follow these steps to set up the project locally.

### 1. Navigate to the Project Directory
Open your terminal or command prompt and navigate to the project folder:
```bash
cd phase2/car_pred
```

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to isolate project dependencies.

**Windows (Command Prompt/PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the required libraries for both the API and the Streamlit application.

```bash
pip install -r api/requirements.txt
pip install -r streamlit_app/requirements.txt
```

---

## How to Run

### 1. Run the Training Pipeline (Optional)
The project comes with pre-trained models in the `model/` directory. However, if you wish to retrain the model from scratch:

```bash
python src/models/train.py
```
This will generate and save the model files (`best_model.joblib`) to the `model/` directory.

### 2. Run the API (Backend)
Start the FastAPI server locally. This handles the prediction logic.

```bash
uvicorn api.main:app --reload
```
- The API will start at: `http://127.0.0.1:8000`
- Interactive Documentation: `http://127.0.0.1:8000/docs`

### 3. Run the Streamlit App (Frontend)
Open a **new terminal window**, activate the virtual environment again, and run the frontend app:

```bash
# Activate venv first (if using a new terminal)
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Run the app
streamlit run streamlit_app/app.py
```

**Important Note on API Connection:**
By default, the Streamlit app might be configured to connect to a deployed API URL. To use your **local** API:
1. Open `streamlit_app/app.py` in a text editor.
2. Locate the `API_URL` variable (around line 9).
3. Change it to your local address:
   ```python
   API_URL = "http://127.0.0.1:8000/predict"
   ```

---

## Troubleshooting

If you encounter issues, try the following solutions:

### 1. `ModuleNotFoundError: No module named ...`
*   **Issue:** Python cannot find the required libraries.
*   **Solution:** Ensure your virtual environment is **activated**. You should see `(venv)` at the start of your command prompt. If active, try installing dependencies again:
    ```bash
    pip install -r api/requirements.txt
    pip install -r streamlit_app/requirements.txt
    ```

### 2. `uvicorn` is not recognized as an internal or external command
*   **Issue:** The system cannot find the `uvicorn` executable.
*   **Solution:** Try running it as a Python module:
    ```bash
    python -m uvicorn api.main:app --reload
    ```

### 3. Streamlit App shows "API connection failed"
*   **Issue:** The frontend cannot communicate with the backend.
*   **Solution:**
    *   Ensure the API is running in a separate terminal window.
    *   Check that the `API_URL` in `streamlit_app/app.py` matches your running API address (usually `http://127.0.0.1:8000/predict`).

### 4. `FileNotFoundError: ❌ No model found`
*   **Issue:** The API cannot find the `.joblib` model files.
*   **Solution:** The `model/` directory might be empty. Run the training script to generate them:
    ```bash
    python src/models/train.py
    ```

### 5. Permission Denied errors
*   **Issue:** Windows sometimes restricts script execution.
*   **Solution:** If using PowerShell, you might need to run:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```

## Project Structure
*   `api/`: Contains the FastAPI backend code (`main.py`).
*   `streamlit_app/`: Contains the Streamlit frontend code (`app.py`).
*   `src/`: Source code for feature engineering and model training.
*   `model/`: Stores the trained machine learning models.
*   `notebooks/`: Jupyter notebooks for data exploration (EDA).
