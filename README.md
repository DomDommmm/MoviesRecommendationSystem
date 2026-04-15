Movie recommendation prototype using matrix factorization (ALS) with QR-based linear solves.

## What is implemented

- ALS for explicit feedback with sparse matrices (`scipy.sparse`).
- Alternating updates for user/item latent factors.
- OLS normal-equation subproblem solved by QR decomposition (`scipy.linalg.qr`).
- Loss tracking for convergence diagnostics.
- Development test on a small slice of MovieLens 1M.

## File

- `QR.py`: end-to-end ALS core and development test runner.

## How to run

1. Put MovieLens 1M `ratings.dat` in one of these paths:
	- `ml-1m/ratings.dat`
	- `ratings.dat`
	- `data/ml-1m/ratings.dat`
	- `MovieLens1M/ratings.dat`
2. Run:

```powershell
python QR.py
```

If no MovieLens file is found, the script automatically runs a synthetic fallback dataset for debugging.

## Notes

- The script slices MovieLens for development (`max_rows=120000`) so iteration speed is suitable while validating math and dimensions.
- You can tune `ALSConfig(factors, reg, iterations)` in `development_test()`.