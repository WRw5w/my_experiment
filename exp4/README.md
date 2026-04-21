# Experiment 4: ADMM Image Denoising

This directory contains the implementation and comparison of the AMDD (ADMM) algorithm for image denoising.

## 📁 Directory Structure

- `ADMM.py`: Main script for running denoising experiments and comparisons.
- `getimage.py`: Utility functions for plotting and saving result visualizations.
- `data/`: Put your custom input images here.
- `outputs/`: All experiment results are stored here, organized by timestamp.
  - `exp_YYYYMMDD_HHMMSS/`:
    - `denoising_performance_log.csv`: Detailed metrics (PSNR, SSIM, Time) for each algorithm and iteration.
    - `denoising_comparison.pdf`: Visualization of denoising results and convergence curves.

## 🚀 How to Run

Run the main experiment:
```bash
python ADMM.py
```

The script will automatically create a new folder in `outputs/` for each run.
