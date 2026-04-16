import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import TimeSeriesSplit
import optuna
import datetime
import pickle
import time

from utils import ALSConfig, ALSExplicitQR

def preprocess(file_path='ratings_filtered.csv'):
    print(f"Đang tải dataset đã lọc: {file_path}...")
    # Load standardized CSV
    df = pd.read_csv(file_path)
    print(f"Tổng số đánh giá sử dụng: {len(df):,}")
    
    # We do not need to sort by Timestamp here again because preprocess.py already strictly ordered it!
    return df


def split_train_test_by_time(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")

    split_idx = int(len(df) * (1.0 - test_size))
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError(f"Invalid time split index {split_idx} for dataframe length {len(df)}.")

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def prepare_fold(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, int, float, dict, dict]:
    """
    1. Creates UserID and MovieID dictionaries based STRICTLY on train_df
    2. Filters out cold-starts from val_df
    3. Calculates global_mean and mean-centers the train ratings
    """
    user_uniques = train_df['UserID'].unique()
    movie_uniques = train_df['MovieID'].unique()
    
    user_map = {uid: idx for idx, uid in enumerate(user_uniques)}
    movie_map = {mid: idx for idx, mid in enumerate(movie_uniques)}
    
    train_mapped = train_df.copy()
    train_mapped['user_idx'] = train_mapped['UserID'].map(user_map).astype(int)
    train_mapped['movie_idx'] = train_mapped['MovieID'].map(movie_map).astype(int)
    
    global_mean = float(train_mapped['Rating'].mean())
    train_mapped['Rating_centered'] = train_mapped['Rating'] - global_mean
    
    num_users = len(user_map)
    num_movies = len(movie_map)
    
    # Filter validation set against cold-starts (users/movies not in train_df)
    val_mapped = val_df[
        val_df['UserID'].isin(user_map) & 
        val_df['MovieID'].isin(movie_map)
    ].copy()
    
    if len(val_mapped) > 0:
        val_mapped['user_idx'] = val_mapped['UserID'].map(user_map).astype(int)
        val_mapped['movie_idx'] = val_mapped['MovieID'].map(movie_map).astype(int)
    else:
        # Fallback if empty
        val_mapped['user_idx'] = pd.Series([], dtype=int)
        val_mapped['movie_idx'] = pd.Series([], dtype=int)
    
    return train_mapped, val_mapped, num_users, num_movies, global_mean, user_map, movie_map


def build_csr_matrix(df: pd.DataFrame, num_users: int, num_movies: int, rating_col: str = 'Rating_centered') -> csr_matrix:
    # CSR (Compressed Sparse Row) tối ưu cho các phép toán đại số tuyến tính trong ALS.
    matrix = csr_matrix(
        (df[rating_col], (df['user_idx'], df['movie_idx'])),
        shape=(num_users, num_movies)
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def rmse_on_dataframe(df: pd.DataFrame, user_factors: np.ndarray, item_factors: np.ndarray, global_mean: float) -> float:
    if len(df) == 0:
        return 0.0

    user_ids = df['user_idx'].to_numpy(dtype=np.int64)
    movie_ids = df['movie_idx'].to_numpy(dtype=np.int64)
    targets = df['Rating'].to_numpy(dtype=np.float64)

    # Dự đoán bằng dot-product cộng lại giá trị trung bình (global mean)
    preds = np.einsum("ij,ij->i", user_factors[user_ids], item_factors[movie_ids]) + global_mean
    
    return float(np.sqrt(np.mean((targets - preds) ** 2)))


def cross_validate_als(df: pd.DataFrame, n_splits: int = 5) -> None:
    tss = TimeSeriesSplit(n_splits=n_splits)
    config = ALSConfig()
    fold_scores: list[float] = []

    print(f"Chạy {n_splits}-Fold Time Series Split với cấu hình mặc định: {config}")

    for fold_idx, (train_idx, val_idx) in enumerate(tss.split(df), start=1):
        train_raw = df.iloc[train_idx]
        val_raw = df.iloc[val_idx]

        train_mapped, val_mapped, num_users, num_movies, global_mean, _, _ = prepare_fold(train_raw, val_raw)

        R_train = build_csr_matrix(train_mapped, num_users, num_movies, rating_col='Rating_centered')
        model = ALSExplicitQR(config)
        user_factors, item_factors = model.fit(R_train)

        train_rmse_centered = model.rmse_observed(R_train)
        val_rmse = rmse_on_dataframe(val_mapped, user_factors, item_factors, global_mean)
        fold_scores.append(val_rmse)

        print(
            f"Fold {fold_idx:02d}: "
            f"train_rmse_cen={train_rmse_centered:.6f} "
            f"val_rmse={val_rmse:.6f} "
            f"train_nnz={R_train.nnz} "
            f"val_nnz={len(val_mapped)}"
        )

    print(f"Validation RMSE mean: {np.mean(fold_scores):.6f}")
    print(f"Validation RMSE std: {np.std(fold_scores):.6f}")


def evaluate_config(
    df: pd.DataFrame,
    config: ALSConfig,
    n_splits: int = 5,
) -> float:
    tss = TimeSeriesSplit(n_splits=n_splits)
    fold_scores: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(tss.split(df), start=1):
        fold_start = time.time()
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}]  --> Training Fold {fold_idx}/{n_splits} on GPU...")
        
        train_raw = df.iloc[train_idx]
        val_raw = df.iloc[val_idx]

        train_mapped, val_mapped, num_users, num_movies, global_mean, _, _ = prepare_fold(train_raw, val_raw)

        if len(val_mapped) == 0:
            continue
            
        R_train = build_csr_matrix(train_mapped, num_users, num_movies, rating_col='Rating_centered')
        model = ALSExplicitQR(config)
        user_factors, item_factors = model.fit(R_train)

        train_rmse_centered = model.rmse_observed(R_train)
        val_rmse = rmse_on_dataframe(val_mapped, user_factors, item_factors, global_mean)
        fold_scores.append(val_rmse)
        
        fold_elapsed = time.time() - fold_start
        print(f"      Fold {fold_idx} completed in {fold_elapsed:.2f}s (RMSE: {val_rmse:.6f})")

    mean_score = float(np.mean(fold_scores))
    return mean_score


def optimize_hyperparameters(df: pd.DataFrame, n_trials: int = 10, n_splits: int = 4) -> ALSConfig:
    def objective(trial: optuna.Trial) -> float:
        config = ALSConfig(
            factors=trial.suggest_categorical("factors", [16, 32, 64, 128]),
            reg_user=trial.suggest_float("reg_user", 1e-4, 0.5, log=True),
            reg_item=trial.suggest_float("reg_item", 1e-4, 0.5, log=True),
            iterations=trial.suggest_int("iterations", 10, 30, step = 5),
            seed=7,
            verbose=False,
        )
        trial_start = time.time()
        score = evaluate_config(df, config, n_splits=n_splits)
        trial_elapsed = time.time() - trial_start
        print(f"Trial {trial.number:02d} completed in {trial_elapsed:.2f}s: score={score:.6f} params={config}")
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
    total_start = time.time()
    # Changed standard file load to point to our newly generated 32M filtered DB!
    df = preprocess('./ratings_filtered.csv')
    
    train_df, test_df = split_train_test_by_time(df, test_size=0.2)

    print(f"Train rows (Raw): {len(train_df)} | Test rows (Raw): {len(test_df)}")

    best_config = optimize_hyperparameters(train_df, n_trials=30)

    print(f"Chạy lại ALS với best config trên bộ Train gốc: {best_config}")
    
    train_mapped, test_mapped, num_users, num_movies, global_mean, user_map, movie_map = prepare_fold(train_df, test_df)
    print(f"Warm-Start Test rows thực tế để đánh giá: {len(test_mapped)} / {len(test_df)}")

    R_train_full = build_csr_matrix(train_mapped, num_users, num_movies, rating_col='Rating_centered')
    final_model = ALSExplicitQR(best_config)
    user_factors, item_factors = final_model.fit(R_train_full)

    final_train_rmse_centered = final_model.rmse_observed(R_train_full)
    test_rmse = rmse_on_dataframe(test_mapped, user_factors, item_factors, global_mean)
    
    print(f"Final train RMSE (centered): {final_train_rmse_centered:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"User factors shape: {user_factors.shape}")
    print(f"Item factors shape: {item_factors.shape}")

    print("\nExporting final model to 'als_model2.pkl'...")
    model_data = {
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_map": user_map,
        "movie_map": movie_map,
        "global_mean": global_mean
    }
    with open('als_model2.pkl', 'wb') as f:
        pickle.dump(model_data, f)
        
    total_elapsed = time.time() - total_start
    print(f"Model saved! Total script execution time: {total_elapsed / 60:.2f} minutes.")


if __name__ == '__main__':
    main()