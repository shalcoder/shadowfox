# Project Setup Guide

For the final project report and results, please see [README.md](README.md).

## Meditron NLP Project Setup

This document contains instructions for setting up and running the Meditron NLP Project.

## Prerequisites

- **Python**: 3.8 or higher.
- **GPU**: NVIDIA GPU with CUDA support is highly recommended.
  - The project uses 4-bit quantization to fit the 7B model into memory.
  - Minimum VRAM: ~6GB (configured in scripts).
- **OS**: Windows (tested), Linux.

## Setup Instructions

1.  **Clone or Download the Project**:
    Ensure you have the project files locally.

2.  **Create a Virtual Environment** (Recommended):
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate
    ```

3.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```

    *Note: If you encounter issues with `bitsandbytes` on Windows, you may need to install a Windows-compatible version:*
    ```powershell
    pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
    ```
    *(Or check for the latest compatible wheel for your specific CUDA version).*

4.  **Verify Setup**:
    Run the check script to ensure the model loads correctly.
    ```powershell
    python check.py
    ```

## Running the Project

### Option 1: Run the Full Analysis Script
This script performs generation tests, clinical note summarization, MCQ evaluation, and a LangChain demo.

**Important**: Run this script from the `scripts` directory so it can correctly locate the `../data` folder.

```powershell
cd scripts
python run_analysis.py
```

Outputs will be saved to `../results/outputs/`.

### Option 2: Interactive Notebook
Explore the analysis step-by-step using Jupyter Notebook.

```powershell
jupyter notebook notebook/meditron_analysis.ipynb
```

## Troubleshooting

### 1. `bitsandbytes` Errors on Windows
**Error**: `ImportError: DLL load failed while importing ...` or warnings about missing CUDA.
**Fix**: The standard `bitsandbytes` package often has issues on Windows.
- Uninstall the standard package: `pip uninstall bitsandbytes`
- Install a Windows-specific wheel (as mentioned in Setup).
- Ensure you have the correct CUDA Toolkit installed (e.g., CUDA 11.8 or 12.x) matching your PyTorch version.

### 2. CUDA Out of Memory (OOM)
**Error**: `CUDA out of memory.`
**Fix**:
- The scripts are configured to use `load_in_4bit=True` to save memory.
- Check `scripts/run_analysis.py` and adjust `max_memory` settings if needed (e.g., reduce "cuda:0" limit).
- Close other GPU-intensive applications.

### 3. File Not Found Errors
**Error**: `FileNotFoundError: [Errno 2] No such file or directory: '../data/...'`
**Fix**:
- Ensure you are running the script from the `scripts/` directory:
  ```powershell
  cd scripts
  python run_analysis.py
  ```

### 4. Slow Performance
- **Cause**: Running on CPU instead of GPU.
- **Fix**: Ensure `torch.cuda.is_available()` returns `True`. If not, reinstall PyTorch with CUDA support:
  ```powershell
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```
  (Replace `cu118` with your CUDA version).

## Dependencies

Key dependencies included in `requirements.txt`:
- `transformers`: For model loading and tokenization.
- `torch`: PyTorch framework.
- `bitsandbytes`: For 4-bit quantization.
- `accelerate`: For efficient model loading.
- `langchain`: For chaining LLM operations.
- `pandas`, `matplotlib`, `seaborn`: For data handling and visualization.
