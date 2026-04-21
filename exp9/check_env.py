import torch
import sys
import os
import subprocess

with open('env_report.txt', 'w') as f:
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Python executable: {sys.executable}\n")
    try:
        f.write(f"Torch version: {torch.__version__}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"GPU count: {torch.cuda.device_count()}\n")
    except Exception as e:
        f.write(f"Torch info error: {e}\n")
    
    try:
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not detected')
        f.write(f"Conda Env: {conda_env}\n")
    except Exception as e:
        f.write(f"Conda info error: {e}\n")
