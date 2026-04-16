# PyTorch GPU Movie Recommendation System

A massive-scale, explicit-feedback movie recommendation engine using Alternating Least Squares (ALS) and QR-based linear solvers. Originally designed for MovieLens 1M, this framework has been completely overhauled to run on **PyTorch tensors**, enabling GPU-accelerated Matrix Factorization capable of handling the **MovieLens 32M** dataset almost effortlessly.

## Overview

The project relies on a modular 3-tier architecture:

- `preprocess.py`: A dedicated script for chronological sorting and core-filtering the raw 32 Million row datasets.
- `models.py`: Handles strict Time-Series validation, prevents "cold-start" overlapping metrics, applies rating mean-centering, and manages Optuna hyperparameter sweeps.
- `utils.py`: The mathematical PyTorch backend. Executes QR linear solves directly on GPU VRAM using zero-copy tensor slicing.

## Features

- **Blazing Fast GPU Acceleration:** Swapped traditional CPU-bound `scipy.sparse` loops for native `torch.Tensor` operations using `torch.linalg.qr`.
- **Zero-Copy Architecture:** The massive 32M Interaction matrix is loaded into VRAM precisely once. Python loops only pass primitive slice coordinates, completely avoiding the overhead of CPU-to-GPU data transfers during model fitting.
- **Core-10 Data Truncation:** Automatically filters the long tail of the bottom 10% interacting users/movies to create dense, mathematically stable matrices.
- **True Chronological Validation:** Uses `TimeSeriesSplit` evaluating models purely on predicting "future interactions" mimicking true production scenarios.
- **Cold-Start Safeguards:** Evaluates metrics exclusively on "warm" overlap entities during validation, preventing the RMSE metric from being artificially inflated by inherently unknown nodes.
- **Optuna Hyperparameter Search:** Automated tuning across latent factors, user/item regularizations, and iteration limits.

## Project Structure

- `preprocess.py`: Data loader, cleaner, and core-n filtering script.
- `models.py`: Data mapping, Optuna search, and final model evaluation.
- `utils.py`: The mathematical core containing the `ALSExplicitQR` engine.

## Requirements

Ensure you are working inside your virtual environment and install the heavily parallelized dependencies:

```bash
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn optuna torch truststore
```

*Note: The project natively requires PyTorch (`torch`). Running the pip install block above will automatically download PyTorch along with its massive suite of underlying CUDA libraries (e.g., `nvidia-cublas`, `nvidia-cusolver`, `triton`). These are the essential drivers that route the Matrix Factorization directly to your GPU VRAM.*

## Data Preparation

Download the [MovieLens 32M Dataset](https://grouplens.org/datasets/movielens/32m/), extract it, and place `ratings.csv` in the `ml-32m` directory (or modify the path in `preprocess.py`).

First, run the preprocessing script to clean and compress the data into a dense matrix format:
```bash
python preprocess.py
```
*This will filter the bottom 10%, drop duplicates, align temporal sequences, and output a clean `ratings_filtered.csv`.*

## How to Run

After preparing the data, trigger the full pipeline to execute the Optuna sweep and train your recommendation model:

```bash
python models.py
```

The script will:
1. Load the preprocessed `ratings_filtered.csv`.
2. Split train/test chronologically (first 80% train, next 20% validation/test).
3. Identify and isolate only "Warm" starting users and movies for accurate algorithm scoring.
4. Scale ratings via global mean-centering to save static dimension states.
5. Ship matrices to your CUDA device.
6. Print final train RMSE and test RMSE.

## Output Interpretation

Because we apply global mean centering prior to PyTorch fitting, your predictions start incredibly close to reality (usually ~3.5 stars) even before learning. 
A highly-tuned Matrix Factorization algorithm on a temporally split explicit MovieLens dataset will typically reach a final Test RMSE somewhere around **0.88 - 0.92**.

## Development Notes

- `utils.py` contains the `ALSConfig` dataclass. If you are experiencing GPU issues, you can explicitly force PyTorch to step down by changing `device: str = "cpu"`.
- If you are running tests to ensure hardware stability, consider reducing `n_trials=50` inside `models.py`'s `main()` function to a lower number like `n_trials=5` to rapidly see the end of the pipeline.