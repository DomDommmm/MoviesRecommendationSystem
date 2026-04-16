from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse
import torch

@dataclass
class ALSConfig:
    factors: int = 32
    reg_user: float = 0.08
    reg_item: float = 0.08
    iterations: int = 15
    seed: int = 7
    verbose: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def solve_linear_qr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve Ax=b using QR decomposition via PyTorch."""
    # mode="reduced" ensures economic QR decomposition (returns small matrices)
    q, r = torch.linalg.qr(a, mode="reduced")
    y = q.T @ b
    # Solve Rx = y using fast upper triangular solver
    x = torch.linalg.solve_triangular(r, y.unsqueeze(1), upper=True).squeeze(1)
    return x

class ALSExplicitQR:
    """Explicit-feedback ALS trained with sparse matrices and QR linear solves on GPU."""

    def __init__(self, config: ALSConfig):
        self.config = config
        self.device = torch.device(self.config.device)
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.loss_history: list[float] = []

    def fit(self, ratings_csr: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        # Tidy up matrix formats
        if not sparse.isspmatrix_csr(ratings_csr):
            ratings_csr = ratings_csr.tocsr()

        ratings_csr.sort_indices()
        n_users, n_items = ratings_csr.shape
        k = self.config.factors

        if n_users == 0 or n_items == 0:
            raise ValueError(f"Empty rating matrix with shape {ratings_csr.shape}.")
        if k <= 0:
            raise ValueError(f"factors must be > 0, got {k}.")

        # Initial random matrices placed directly onto designated device
        torch.manual_seed(self.config.seed)
        x = torch.randn(n_users, k, dtype=torch.float64, device=self.device) * 0.1
        y = torch.randn(n_items, k, dtype=torch.float64, device=self.device) * 0.1

        # We keep indptr on CPU for fast python looping slice boundaries, 
        # but load indices/data payload arrays fully into memory/VRAM once.
        csr_indptr = ratings_csr.indptr
        csr_indices = torch.from_numpy(ratings_csr.indices).to(self.device, non_blocking=True)
        csr_data = torch.from_numpy(ratings_csr.data).to(self.device, non_blocking=True)

        ratings_csc = ratings_csr.tocsc()
        csc_indptr = ratings_csc.indptr
        csc_indices = torch.from_numpy(ratings_csc.indices).to(self.device, non_blocking=True)
        csc_data = torch.from_numpy(ratings_csc.data).to(self.device, non_blocking=True)

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

        # Returns numpy arrays to avoid API breakage in main file
        self.user_factors = x.cpu().numpy()
        self.item_factors = y.cpu().numpy()
        return self.user_factors, self.item_factors

    def _update_factors(
        self,
        n_entities: int,
        k: int,
        indptr: np.ndarray,
        indices: torch.Tensor,
        data: torch.Tensor,
        other_factors: torch.Tensor,
        reg: float
    ) -> torch.Tensor:
        out = torch.zeros((n_entities, k), dtype=torch.float64, device=self.device)
        identity = torch.eye(k, dtype=torch.float64, device=self.device)

        for u in range(n_entities):
            start, end = int(indptr[u]), int(indptr[u + 1])
            if start == end:
                continue

            # Pure GPU array slices (no memory copies generated)
            target_ids = indices[start:end]
            values = data[start:end]

            y_u = other_factors[target_ids]
            
            # Massive batched math happens squarely on hardware compute cores
            a = y_u.T @ y_u + reg * target_ids.shape[0] * identity
            b = y_u.T @ values
            out[u] = solve_linear_qr(a, b)

        return out

    def compute_loss(
        self,
        ratings_csr: sparse.csr_matrix,
        x: torch.Tensor,
        y: torch.Tensor,
        reg_user: float,
        reg_item: float,
    ) -> float:
        # Pushing COE to compute unified device loss avoiding cpu bottlenecks
        coo = ratings_csr.tocoo(copy=False)
        rows_t = torch.from_numpy(coo.row).to(self.device, non_blocking=True)
        cols_t = torch.from_numpy(coo.col).to(self.device, non_blocking=True)
        data_t = torch.from_numpy(coo.data).to(self.device, non_blocking=True)

        preds = (x[rows_t] * y[cols_t]).sum(dim=1)
        residual = data_t - preds
        data_loss = float((residual * residual).sum().item())
        reg_loss = float(reg_user * (x * x).sum().item() + reg_item * (y * y).sum().item())
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
    print(f"PyTorch using hardware device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
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
