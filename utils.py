from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse
import scipy.linalg
import torch
from joblib import Parallel, delayed

@dataclass
class ALSConfig:
    factors: int = 32
    reg_user: float = 0.08
    reg_item: float = 0.08
    iterations: int = 15
    seed: int = 7
    verbose: bool = True
    device: str = "cpu"  # Legacy field

def solve_linear_qr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Legacy PyTorch GPU QR Decomposition Solver."""
    q, r = torch.linalg.qr(a, mode="reduced")
    y = q.T @ b
    x = torch.linalg.solve_triangular(r, y.unsqueeze(1), upper=True).squeeze(1)
    return x

def solve_linear_cholesky(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax=b using Cholesky decomposition via SciPy LAPACK."""
    try:
        c, lower = scipy.linalg.cho_factor(a)
        return scipy.linalg.cho_solve((c, lower), b)
    except scipy.linalg.LinAlgError:
        # Fallback for numerically unstable matrices
        return np.linalg.lstsq(a, b, rcond=None)[0]

def _process_chunk(
    chunk_start: int,
    chunk_end: int,
    n_entities: int,
    k: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    other_factors: np.ndarray,
    reg: float
) -> np.ndarray:
    actual_end = min(chunk_end, n_entities)
    out = np.zeros((actual_end - chunk_start, k), dtype=np.float64)
    identity = np.eye(k, dtype=np.float64)
    
    for i, u in enumerate(range(chunk_start, actual_end)):
        start, end = indptr[u], indptr[u + 1]
        if start == end:
            continue
            
        target_ids = indices[start:end]
        values = data[start:end]
        
        y_u = other_factors[target_ids]
        
        a = y_u.T @ y_u + reg * target_ids.shape[0] * identity
        b = y_u.T @ values
        
        out[i] = solve_linear_cholesky(a, b)
        
    return out

class ALSExplicitQR:
    """Explicit-feedback ALS trained using Multi-core joblib and NumPy Cholesky linear solves on CPU."""

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

        # Initial random matrices placed directly into host memory
        np.random.seed(self.config.seed)
        x = np.random.normal(0, 0.1, (n_users, k))
        y = np.random.normal(0, 0.1, (n_items, k))

        csr_indptr = ratings_csr.indptr
        csr_indices = ratings_csr.indices
        csr_data = ratings_csr.data

        ratings_csc = ratings_csr.tocsc()
        csc_indptr = ratings_csc.indptr
        csc_indices = ratings_csc.indices
        csc_data = ratings_csc.data

        self.loss_history = []

        for epoch in range(1, self.config.iterations + 1):
            x = self._update_factors(
                n_users, k, csr_indptr, csr_indices, csr_data, y, self.config.reg_user
            )
            y = self._update_factors(
                n_items, k, csc_indptr, csc_indices, csc_data, x, self.config.reg_item
            )

            loss = self.compute_loss(ratings_csr, x, y, self.config.reg_user, self.config.reg_item)
            self.loss_history.append(loss)

            if self.config.verbose:
                delta = 0.0 if epoch == 1 else self.loss_history[-2] - self.loss_history[-1]
                print(f"epoch={epoch:02d} loss={loss:.6f} delta={delta:.6f}")

        self.user_factors = x
        self.item_factors = y
        return self.user_factors, self.item_factors

    def _update_factors(
        self,
        n_entities: int,
        k: int,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        other_factors: np.ndarray,
        reg: float
    ) -> np.ndarray:
        n_jobs = -1  # Use all available logical CPU cores exactly
        chunk_size = max(1, n_entities // 128)  # Partition into rough batch blocks
        
        chunks = [(i, i + chunk_size) for i in range(0, n_entities, chunk_size)]
        
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_chunk)(
                start, end, n_entities, k, indptr, indices, data, other_factors, reg
            ) for start, end in chunks
        )
        
        out = np.vstack(results)
        return out

    def compute_loss(
        self,
        ratings_csr: sparse.csr_matrix,
        x: np.ndarray,
        y: np.ndarray,
        reg_user: float,
        reg_item: float,
        batch_size: int = 2_000_000
    ) -> float:
        """
        OOM-Safe Batched Loss Computation.
        Processes the massive CSR matrix in manageable chunks to prevent RAM spikes.
        """
        coo = ratings_csr.tocoo(copy=False)
        n_elements = coo.data.shape[0]
        data_loss = 0.0

        for start_idx in range(0, n_elements, batch_size):
            end_idx = min(start_idx + batch_size, n_elements)
            
            r_batch = coo.row[start_idx:end_idx]
            c_batch = coo.col[start_idx:end_idx]
            d_batch = coo.data[start_idx:end_idx]

            preds = np.einsum("ij,ij->i", x[r_batch], y[c_batch])
            residual = d_batch - preds
            data_loss += float(np.sum(residual ** 2))

        reg_loss = float(reg_user * np.sum(x ** 2) + reg_item * np.sum(y ** 2))
        return data_loss + reg_loss

    def rmse_observed(self, ratings_csr: sparse.csr_matrix) -> float:
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model must be fitted before calling rmse_observed().")
        
        # Applying the same batching logic to RMSE to keep it completely OOM-safe
        coo = ratings_csr.tocoo(copy=False)
        n_elements = coo.data.shape[0]
        batch_size = 2_000_000
        total_sq_error = 0.0

        for start_idx in range(0, n_elements, batch_size):
            end_idx = min(start_idx + batch_size, n_elements)
            
            r_batch = coo.row[start_idx:end_idx]
            c_batch = coo.col[start_idx:end_idx]
            d_batch = coo.data[start_idx:end_idx]
            
            preds = np.einsum("ij,ij->i", self.user_factors[r_batch], self.item_factors[c_batch])
            total_sq_error += float(np.sum((d_batch - preds) ** 2))

        mse = total_sq_error / n_elements
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
    print(f"NumPy Backend Activated with Multiprocessing!")
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

    config = ALSConfig(factors=32, reg_user=0.08, reg_item=0.08, iterations=12, seed=7, verbose=True)
    model = ALSExplicitQR(config)
    x, y = model.fit(ratings)

    if x.shape[1] != config.factors or y.shape[1] != config.factors:
        raise RuntimeError(f"Factor dimensions are wrong: {x.shape=} and {y.shape=} with {config.factors=}")

    initial_loss = model.loss_history[0]
    final_loss = model.loss_history[-1]
    print(f"initial_loss={initial_loss:.6f} final_loss={final_loss:.6f}")
    print(f"observed_rmse={model.rmse_observed(ratings):.6f}")

if __name__ == "__main__":
    development_test()
