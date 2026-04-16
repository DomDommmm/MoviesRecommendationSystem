# Movie Recommendation System

Matrix factorization prototype for explicit-feedback movie recommendation using Alternating Least Squares (ALS) with QR-based linear solves.

## Overview

The project now contains two main layers:

- `utils.py`: the mathematical ALS core.
- `models.py`: preprocessing, time-based validation, Optuna tuning, and final evaluation.

The pipeline uses MovieLens 1M ratings with timestamps, so validation is chronological instead of random K-Fold.

## Features

- Sparse matrix training with `scipy.sparse`.
- ALS alternating updates for user and item factors.
- QR decomposition for solving the normal equations.
- Separate user and item regularization terms.
- Time-series split for validation when timestamps are available.
- Optuna hyperparameter search.
- Final train and held-out test RMSE reporting.

## Project Structure

- `utils.py`: ALS implementation, QR solver, and loss computation.
- `models.py`: data loading, chronological split, Optuna search, and final model training.
- `ml-1m/ratings.dat`: MovieLens 1M ratings file expected by the scripts.

## Requirements

Use the project virtual environment and install the main dependencies:

```bash
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn optuna truststore
```

Notes:
- `truststore` is used so dataset downloads work in environments with custom certificate chains.
- The project was validated on Python 3.12.

## Data Preparation

The code expects MovieLens 1M `ratings.dat` in one of these locations:

- `./ml-1m/ratings.dat`
- `./ratings.dat`
- `./data/ml-1m/ratings.dat`
- `./MovieLens1M/ratings.dat`

`models.py` reads the file, creates `user_idx` and `movie_idx`, sorts rows by `Timestamp`, then splits the data chronologically:

- first 80%: train
- last 20%: test

## How to Run

Run the full pipeline from `models.py`:

```bash
python models.py
```

The script will:

1. Load and preprocess MovieLens ratings.
2. Split train/test by timestamp.
3. Run Optuna on the training portion using `TimeSeriesSplit`.
4. Retrain ALS with the best hyperparameters.
5. Print final train RMSE and test RMSE.

## ALS Configuration

The current ALS config supports:

- `factors`: latent dimension
- `reg_user`: user-factor regularization
- `reg_item`: item-factor regularization
- `iterations`: number of alternating update steps
- `seed`: random seed
- `verbose`: training logs

## Hyperparameter Search

`models.py` uses Optuna to search over:

- `factors`
- `reg_user`
- `reg_item`
- `iterations`

Validation is done with `TimeSeriesSplit`, so each fold respects chronological order.

## Output Interpretation

When you run the training script, you will see:

- training and validation RMSE per fold
- the best Optuna trial
- final train RMSE after retraining on the full training set
- test RMSE on the held-out chronological test split

Important:
- `Final train RMSE` is measured on the data used to refit the final model.
- `Test RMSE` is the number to use for real model quality comparison.

## Development Notes

- `utils.py` focuses only on the ALS math core and should stay model-agnostic.
- `models.py` is the right place for experiment logic, splits, and tuning.
- For faster debugging, reduce `n_trials` in `models.py` before running full Optuna search.

## Example Workflow

```bash
source .venv/bin/activate
python models.py
```

If you want a quicker experiment first, reduce the Optuna trial count inside `main()`.