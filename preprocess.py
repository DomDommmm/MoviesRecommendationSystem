import pandas as pd

print("Loading raw ML-32M dataset...")
df = pd.read_csv('ml-32m/ratings.csv')
print(f"Raw data: {len(df):,} ratings")

# Drop completely missing rows
df = df.dropna()

# Enforce reasonable rating bounds
df = df[df['rating'].between(0.5, 5.0)]

# Sort by time to ensure later Time Series CV is correctly representing Past -> Future
df = df.sort_values('timestamp')

# Drop duplicated clicks keeping the most recent interaction
df = df.drop_duplicates(subset=['userId', 'movieId'], keep='last')

# ─────────────────────────────────────────────
# REMOVE LONG-TAIL SPARSITY (10% LOW ACTIVITY)
# ─────────────────────────────────────────────
print("Filtering bottom 10% users and movies...")
user_counts = df['userId'].value_counts()
user_threshold = user_counts.quantile(0.10)
df = df[df['userId'].isin(user_counts[user_counts > user_threshold].index)]

movie_counts = df['movieId'].value_counts()
movie_threshold = movie_counts.quantile(0.10)
df = df[df['movieId'].isin(movie_counts[movie_counts > movie_threshold].index)]

print("\n=== AFTER FILTERING ===")
print(f"Ratings      : {len(df):,}")
print(f"Users        : {df['userId'].nunique():,}")
print(f"Movies       : {df['movieId'].nunique():,}")

# ─────────────────────────────────────────────
# EXPORT PRE-PROCESSED CLEAN DATASET
# (No Train/Test random splitting happens here! models.py takes care of that)
# ─────────────────────────────────────────────
# Capitalize columns so our models.py pipeline implicitly recognizes them.
df.rename(columns={
    'userId': 'UserID', 
    'movieId': 'MovieID', 
    'rating': 'Rating', 
    'timestamp': 'Timestamp'
}, inplace=True)

output_file = "ratings_filtered.csv"
print(f"\nSaving to {output_file}...")
df.to_csv(output_file, index=False)
print("Done! You can now load this file natively in your models.")
