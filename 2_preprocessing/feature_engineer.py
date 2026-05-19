"""
Step 2b: Engineer features for ranking model.
Builds user embeddings (ALS), session context features,
and temporal engagement signals.

Run: python feature_engineer.py
Input:  data/sessions.csv, data/session_arcs.csv
Output: data/features.parquet
"""

import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit
import joblib
import os

DATA_DIR = "../data"
MODELS_DIR = "../models"
os.makedirs(MODELS_DIR, exist_ok=True)

MOOD_FEATURES = ["valence", "energy", "danceability", "acousticness", "tempo"]
EMBEDDING_DIM = 32


def build_user_item_matrix(df):
    """Build user-item interaction matrix weighted by play count."""
    plays = df.groupby(["user", "track"]).size().reset_index(name="plays")
    user_idx = {u: i for i, u in enumerate(plays["user"].unique())}
    item_idx = {t: i for i, t in enumerate(plays["track"].unique())}

    rows = plays["user"].map(user_idx)
    cols = plays["track"].map(item_idx)
    data = plays["plays"].values

    matrix = sparse.csr_matrix((data, (rows, cols)),
                                shape=(len(user_idx), len(item_idx)))
    return matrix, user_idx, item_idx


def train_als(matrix):
    """Train ALS collaborative filtering to get user/item embeddings."""
    model = implicit.als.AlternatingLeastSquares(
        factors=EMBEDDING_DIM,
        regularization=0.1,
        iterations=30,
        random_state=42,
    )
    model.fit(matrix)
    return model


def engineer_track_features(df, session_arcs):
    """Build per-track features including popularity and mood stats."""
    track_feats = df.groupby("track").agg(
        play_count=("user", "count"),
        unique_users=("user", "nunique"),
        mean_valence=("valence", "mean"),
        mean_energy=("energy", "mean"),
        mean_danceability=("danceability", "mean"),
        mean_acousticness=("acousticness", "mean"),
        mean_tempo=("tempo", "mean"),
    ).reset_index()
    track_feats["log_play_count"] = np.log1p(track_feats["play_count"])
    return track_feats


def engineer_session_features(session_arcs):
    """Add temporal and contextual session features."""
    arcs = session_arcs.copy()
    arcs["is_weekend"]    = (arcs["day_of_week"] >= 5).astype(int)
    arcs["is_late_night"] = (arcs["hour_of_day"].between(22, 23) | arcs["hour_of_day"].between(0, 4)).astype(int)
    arcs["is_morning"]    = arcs["hour_of_day"].between(6, 10).astype(int)
    arcs["is_evening"]    = arcs["hour_of_day"].between(18, 21).astype(int)
    return arcs


def main():
    df   = pd.read_csv(f"{DATA_DIR}/sessions.csv")
    arcs = pd.read_csv(f"{DATA_DIR}/session_arcs.csv")
    print(f"Loaded {len(df):,} session tracks, {df['user'].nunique()} users")

    # user/item embeddings via ALS
    print("Building user-item matrix...")
    matrix, user_idx, item_idx = build_user_item_matrix(df)
    print(f"Matrix shape: {matrix.shape} | density: {matrix.nnz / (matrix.shape[0]*matrix.shape[1]):.4%}")

    print("Training ALS embeddings...")
    als_model = train_als(matrix)
    joblib.dump(als_model, f"{MODELS_DIR}/als_model.pkl")
    joblib.dump(user_idx,  f"{MODELS_DIR}/user_idx.pkl")
    joblib.dump(item_idx,  f"{MODELS_DIR}/item_idx.pkl")
    print(f"ALS trained. User embeddings: {als_model.user_factors.shape}")

    # track features
    track_feats   = engineer_track_features(df, arcs)
    session_feats = engineer_session_features(arcs)

    track_feats.to_parquet(f"{DATA_DIR}/track_features.parquet", index=False)
    session_feats.to_parquet(f"{DATA_DIR}/session_features.parquet", index=False)

    # attach ALS user embedding to sessions
    user_emb_df = pd.DataFrame(
        als_model.user_factors,
        index=list(user_idx.keys()),
        columns=[f"user_emb_{i}" for i in range(EMBEDDING_DIM)],
    ).reset_index().rename(columns={"index": "user"})
    user_emb_df.to_parquet(f"{DATA_DIR}/user_embeddings.parquet", index=False)

    print("Feature engineering complete.")
    print(f"  track_features.parquet : {len(track_feats):,} tracks")
    print(f"  session_features.parquet: {len(session_feats):,} sessions")
    print(f"  user_embeddings.parquet : {len(user_emb_df):,} users")


if __name__ == "__main__":
    main()
