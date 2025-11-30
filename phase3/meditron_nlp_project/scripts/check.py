# scripts/check.py
import sys
import importlib
import psutil


pkgs = ["transformers", "torch", "bitsandbytes", "pandas"]
print("Python:", sys.version)
for p in pkgs:
 try:
   m = importlib.import_module(p)
   print(p, getattr(m, "__version__", "version unknown"))
 except Exception as e:
    print(p, "NOT INSTALLED (", e, ")")


# GPU info
try:
  import torch
  print("torch CUDA available:", torch.cuda.is_available())
  if torch.cuda.is_available():
     print("GPU name:", torch.cuda.get_device_name(0))
     print("VRAM(GB):", torch.cuda.get_device_properties(0).total_memory/1024**3)
except Exception as e:
     print("Torch GPU check failed:", e)


# memory
vm = psutil.virtual_memory()
print(f"Total RAM: {vm.total/1024**3:.2f} GB, Free: {vm.available/1024**3:.2f} GB")