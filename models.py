import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import TimeSeriesSplit
import optuna

from utils import ALSConfig, ALSExplicitQR

def preprocess(file_path='ratings.dat'):
    # MovieLens-1M dùng dâu '::' để phân cách cột
    col_names = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    df = pd.read_csv(file_path, sep='::', engine='python', names=col_names)
    
    print(f"Tổng số đánh giá ban đầu: {len(df)}")

    df['user_idx'] = df['UserID'].astype('category').cat.codes
    df['movie_idx'] = df['MovieID'].astype('category').cat.codes

    # Sort by time so any later split respects the original rating order.
    df = df.sort_values('Timestamp', kind='mergesort').reset_index(drop=True)
    
    num_users = df['user_idx'].nunique()
    num_movies = df['movie_idx'].nunique()
    print(f"Số Users thực tế: {num_users} | Số Movies thực tế: {num_movies}")

    print("Hoàn tất!")

    return df, num_users, num_movies


def split_train_test_by_time(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")

    split_idx = int(len(df) * (1.0 - test_size))
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError(f"Invalid time split index {split_idx} for dataframe length {len(df)}.")

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def build_csr_matrix(df: pd.DataFrame, num_users: int, num_movies: int) -> csr_matrix:
    # CSR (Compressed Sparse Row) tối ưu cho các phép toán đại số tuyến tính trong ALS.
    matrix = csr_matrix(
        (df['Rating'], (df['user_idx'], df['movie_idx'])),
        shape=(num_users, num_movies)
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def rmse_on_dataframe(df: pd.DataFrame, user_factors: np.ndarray, item_factors: np.ndarray) -> float:
    user_ids = df['user_idx'].to_numpy(dtype=np.int64)
    movie_ids = df['movie_idx'].to_numpy(dtype=np.int64)
    targets = df['Rating'].to_numpy(dtype=np.float64)

    preds = np.einsum("ij,ij->i", user_factors[user_ids], item_factors[movie_ids])
    return float(np.sqrt(np.mean((targets - preds) ** 2)))


def cross_validate_als(df: pd.DataFrame, num_users: int, num_movies: int, n_splits: int = 5) -> None:
    tss = TimeSeriesSplit(n_splits=n_splits)
    config = ALSConfig()
    fold_scores: list[float] = []

    print(f"Chạy {n_splits}-Fold Time Series Split với cấu hình mặc định: {config}")

    for fold_idx, (train_idx, val_idx) in enumerate(tss.split(df), start=1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        R_train = build_csr_matrix(train_df, num_users, num_movies)
        model = ALSExplicitQR(config)
        user_factors, item_factors = model.fit(R_train)

        train_rmse = model.rmse_observed(R_train)
        val_rmse = rmse_on_dataframe(val_df, user_factors, item_factors)
        fold_scores.append(val_rmse)

        print(
            f"Fold {fold_idx:02d}: "
            f"train_rmse={train_rmse:.6f} "
            f"val_rmse={val_rmse:.6f} "
            f"train_nnz={R_train.nnz}"
        )

    print(f"Validation RMSE mean: {np.mean(fold_scores):.6f}")
    print(f"Validation RMSE std: {np.std(fold_scores):.6f}")


def evaluate_config(
    df: pd.DataFrame,
    num_users: int,
    num_movies: int,
    config: ALSConfig,
    n_splits: int = 5,
) -> float:
    tss = TimeSeriesSplit(n_splits=n_splits)
    fold_scores: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(tss.split(df), start=1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        R_train = build_csr_matrix(train_df, num_users, num_movies)
        model = ALSExplicitQR(config)
        user_factors, item_factors = model.fit(R_train)

        train_rmse = model.rmse_observed(R_train)
        val_rmse = rmse_on_dataframe(val_df, user_factors, item_factors)
        fold_scores.append(val_rmse)

        print(
            f"Fold {fold_idx:02d}: "
            f"train_rmse={train_rmse:.6f} "
            f"val_rmse={val_rmse:.6f} "
            f"train_nnz={R_train.nnz}"
        )

    mean_score = float(np.mean(fold_scores))
    return mean_score


def optimize_hyperparameters(df: pd.DataFrame, num_users: int, num_movies: int, n_trials: int = 10, n_splits: int = 5) -> ALSConfig:
    def objective(trial: optuna.Trial) -> float:
        config = ALSConfig(
            factors=trial.suggest_int("factors", 8, 32, step=8),
            reg_user=trial.suggest_float("reg_user", 1e-4, 0.5, log=True),
            reg_item=trial.suggest_float("reg_item", 1e-4, 0.5, log=True),
            iterations=trial.suggest_int("iterations", 5, 20),
            seed=7,
            verbose=False,
        )
        score = evaluate_config(df, num_users, num_movies, config, n_splits=n_splits)
        print(
            f"Trial {trial.number:02d}: "
            f"score={score:.6f} "
            f"params={config}"
        )
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_config = ALSConfig(
        factors=int(best_params["factors"]),
        reg_user=float(best_params["reg_user"]),
        reg_item=float(best_params["reg_item"]),
        iterations=int(best_params["iterations"]),
        seed=7,
        verbose=True,
    )

    print("Best trial:")
    print(f"  value={study.best_value:.6f}")
    print(f"  params={study.best_params}")
    return best_config


def main() -> None:
    df, num_users, num_movies = preprocess('./ml-1m/ratings.dat')
    train_df, test_df = split_train_test_by_time(df, test_size=0.2)

    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    best_config = optimize_hyperparameters(train_df, num_users, num_movies, n_trials=50, n_splits=5)

    print(f"Chạy lại ALS với best config: {best_config}")
    R_train_full = build_csr_matrix(train_df, num_users, num_movies)
    final_model = ALSExplicitQR(best_config)
    user_factors, item_factors = final_model.fit(R_train_full)

    final_train_rmse = final_model.rmse_observed(R_train_full)
    test_rmse = rmse_on_dataframe(test_df, user_factors, item_factors)
    print(f"Final train RMSE: {final_train_rmse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"User factors shape: {user_factors.shape}")
    print(f"Item factors shape: {item_factors.shape}")


if __name__ == '__main__':
    main()