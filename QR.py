from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.linalg import qr, solve_triangular


@dataclass
class ALSConfig:
    factors: int = 32
    reg: float = 0.08
    iterations: int = 15
    seed: int = 7
    verbose: bool = True


def solve_linear_qr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax=b using QR decomposition, where A is square and full rank."""
    if a.ndim != 2 or b.ndim != 1:
        raise ValueError(f"Expected a.ndim=2 and b.ndim=1, got {a.ndim=} and {b.ndim=}")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"A must be square. Got shape {a.shape}.")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Incompatible dimensions for Ax=b. Got {a.shape=} and {b.shape=}.")

    q, r = qr(a, mode="economic", overwrite_a=False, check_finite=False)
    y = q.T @ b
    x = solve_triangular(r, y, lower=False, check_finite=False)
    return x


class ALSExplicitQR:
    """Explicit-feedback ALS trained with sparse matrices and QR linear solves."""

    def __init__(self, config: ALSConfig):
        self.config = config
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.loss_history: list[float] = []

    def fit(self, ratings_csr: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        if not sparse.isspmatrix_csr(ratings_csr):
            ratings_csr = ratings_csr.tocsr()

        ratings_csr.sort_indices()
        n_users, n_items = ratings_csr.shape
        k = self.config.factors

        if n_users == 0 or n_items == 0:
            raise ValueError(f"Empty rating matrix with shape {ratings_csr.shape}.")
        if k <= 0:
            raise ValueError(f"factors must be > 0, got {k}.")

        rng = np.random.default_rng(self.config.seed)
        x = rng.normal(0.0, 0.1, size=(n_users, k))
        y = rng.normal(0.0, 0.1, size=(n_items, k))

        ratings_csc = ratings_csr.tocsc()
        self.loss_history = []

        for epoch in range(1, self.config.iterations + 1):
            x = self._update_user_factors(ratings_csr, y)
            y = self._update_item_factors(ratings_csc, x)

            loss = self.compute_loss(ratings_csr, x, y, reg=self.config.reg)
            self.loss_history.append(loss)

            if self.config.verbose:
                delta = 0.0 if epoch == 1 else self.loss_history[-2] - self.loss_history[-1]
                print(f"epoch={epoch:02d} loss={loss:.6f} delta={delta:.6f}")

        self.user_factors = x
        self.item_factors = y
        return x, y

    def _update_user_factors(self, ratings_csr: sparse.csr_matrix, item_factors: np.ndarray) -> np.ndarray:
        n_users = ratings_csr.shape[0]
        k = self.config.factors
        out = np.zeros((n_users, k), dtype=np.float64)
        identity = np.eye(k, dtype=np.float64)

        for u in range(n_users):
            start, end = ratings_csr.indptr[u], ratings_csr.indptr[u + 1]
            item_ids = ratings_csr.indices[start:end]
            values = ratings_csr.data[start:end]

            if item_ids.size == 0:
                continue

            y_u = item_factors[item_ids]
            if y_u.shape[1] != k:
                raise ValueError(
                    f"Dimension mismatch in user update: {y_u.shape[1]=} but factors={k}."
                )

            a = y_u.T @ y_u + self.config.reg * item_ids.size * identity
            b = y_u.T @ values
            out[u] = solve_linear_qr(a, b)

        return out

    def _update_item_factors(self, ratings_csc: sparse.csc_matrix, user_factors: np.ndarray) -> np.ndarray:
        n_items = ratings_csc.shape[1]
        k = self.config.factors
        out = np.zeros((n_items, k), dtype=np.float64)
        identity = np.eye(k, dtype=np.float64)

        for i in range(n_items):
            start, end = ratings_csc.indptr[i], ratings_csc.indptr[i + 1]
            user_ids = ratings_csc.indices[start:end]
            values = ratings_csc.data[start:end]

            if user_ids.size == 0:
                continue

            x_i = user_factors[user_ids]
            if x_i.shape[1] != k:
                raise ValueError(
                    f"Dimension mismatch in item update: {x_i.shape[1]=} but factors={k}."
                )

            a = x_i.T @ x_i + self.config.reg * user_ids.size * identity
            b = x_i.T @ values
            out[i] = solve_linear_qr(a, b)

        return out

    @staticmethod
    def compute_loss(ratings_csr: sparse.csr_matrix, x: np.ndarray, y: np.ndarray, reg: float | None = None) -> float:
        if reg is None:
            reg = 0.0

        coo = ratings_csr.tocoo(copy=False)
        preds = np.einsum("ij,ij->i", x[coo.row], y[coo.col])
        residual = coo.data - preds
        data_loss = float(residual @ residual)
        reg_loss = float(reg * (np.sum(x * x) + np.sum(y * y)))
        return data_loss + reg_loss

    def rmse_observed(self, ratings_csr: sparse.csr_matrix) -> float:
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model must be fitted before calling rmse_observed().")
        coo = ratings_csr.tocoo(copy=False)
        preds = np.einsum("ij,ij->i", self.user_factors[coo.row], self.item_factors[coo.col])
        mse = np.mean((coo.data - preds) ** 2)
        return float(np.sqrt(mse))


def _find_first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_movielens_1m_slice(
    ratings_path: str | Path,
    max_rows: int = 100_000,
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
) -> sparse.csr_matrix:
    """
    Load a small development slice from MovieLens 1M ratings.dat.

    File format per line: userId::movieId::rating::timestamp
    """
    ratings_path = Path(ratings_path)
    if not ratings_path.exists():
        raise FileNotFoundError(f"File not found: {ratings_path}")

    user_ids: list[int] = []
    item_ids: list[int] = []
    values: list[float] = []

    with ratings_path.open("r", encoding="latin-1") as f:
        for idx, line in enumerate(f):
            if idx >= max_rows:
                break
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            user_ids.append(int(parts[0]))
            item_ids.append(int(parts[1]))
            values.append(float(parts[2]))

    if not values:
        raise ValueError("No rating rows were loaded. Check max_rows and file format.")

    u_raw = np.asarray(user_ids, dtype=np.int64)
    i_raw = np.asarray(item_ids, dtype=np.int64)
    r = np.asarray(values, dtype=np.float64)

    # Optional sparsity cleanup for more stable training on tiny slices.
    u_unique, u_idx = np.unique(u_raw, return_inverse=True)
    i_unique, i_idx = np.unique(i_raw, return_inverse=True)

    u_count = np.bincount(u_idx)
    i_count = np.bincount(i_idx)
    keep_mask = (u_count[u_idx] >= min_user_ratings) & (i_count[i_idx] >= min_item_ratings)

    if keep_mask.sum() == 0:
        raise ValueError("All rows were filtered out; lower min_user_ratings/min_item_ratings.")

    u_idx = u_idx[keep_mask]
    i_idx = i_idx[keep_mask]
    r = r[keep_mask]

    _, u_idx = np.unique(u_idx, return_inverse=True)
    _, i_idx = np.unique(i_idx, return_inverse=True)

    n_users = int(u_idx.max()) + 1
    n_items = int(i_idx.max()) + 1

    ratings = sparse.coo_matrix((r, (u_idx, i_idx)), shape=(n_users, n_items)).tocsr()
    ratings.sum_duplicates()
    ratings.sort_indices()
    return ratings


def build_synthetic_small() -> sparse.csr_matrix:
    """Fallback dataset for debugging when MovieLens file is unavailable."""
    rng = np.random.default_rng(123)
    n_users, n_items, k = 120, 90, 8

    true_u = rng.normal(size=(n_users, k))
    true_i = rng.normal(size=(n_items, k))

    nnz = 1800
    rows = rng.integers(0, n_users, size=nnz)
    cols = rng.integers(0, n_items, size=nnz)
    vals = np.einsum("ij,ij->i", true_u[rows], true_i[cols]) + 0.1 * rng.normal(size=nnz)

    vals = np.clip(vals + 3.5, 0.5, 5.0)
    ratings = sparse.coo_matrix((vals, (rows, cols)), shape=(n_users, n_items)).tocsr()
    ratings.sum_duplicates()
    ratings.sort_indices()
    return ratings


def development_test() -> None:
    candidates = [
        Path("ml-1m/ratings.dat"),
        Path("ratings.dat"),
        Path("data/ml-1m/ratings.dat"),
        Path("MovieLens1M/ratings.dat"),
    ]
    path = _find_first_existing(candidates)

    if path is None:
        print("MovieLens 1M file not found. Running synthetic fallback dataset.")
        ratings = build_synthetic_small()
    else:
        print(f"Loading MovieLens 1M slice from: {path}")
        ratings = load_movielens_1m_slice(path, max_rows=120_000)

    print(f"ratings shape={ratings.shape} nnz={ratings.nnz} density={ratings.nnz / (ratings.shape[0] * ratings.shape[1]):.6f}")

    config = ALSConfig(factors=32, reg=0.08, iterations=12, seed=7, verbose=True)
    model = ALSExplicitQR(config)
    x, y = model.fit(ratings)

    if x.shape[1] != config.factors or y.shape[1] != config.factors:
        raise RuntimeError(
            f"Factor dimensions are wrong: {x.shape=} and {y.shape=} with {config.factors=}"
        )

    initial_loss = model.loss_history[0]
    final_loss = model.loss_history[-1]
    print(f"initial_loss={initial_loss:.6f} final_loss={final_loss:.6f}")
    print(f"observed_rmse={model.rmse_observed(ratings):.6f}")

    if not np.isfinite(final_loss):
        raise RuntimeError("Loss became non-finite. Check matrix construction and linear solves.")

    if final_loss > initial_loss:
        print("Warning: loss increased overall. Consider tuning reg/factors/iterations.")


if __name__ == "__main__":
    development_test()
