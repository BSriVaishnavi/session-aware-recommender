"""
Step 3c: LightGBM ranking model.
Combines ALS candidate generation + trajectory-conditioned ranking.
Trains on (session_context, candidate_track, label) tuples.

Run: python lightgbm_ranker.py
Input:  data/sessions.csv, data/session_arcs_clustered.csv,
        data/track_features.parquet, data/user_embeddings.parquet
Output: models/lgbm_ranker.pkl
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import implicit
import mlflow
import os
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR   = "../data"
MODELS_DIR = "../models"
N_CANDIDATES = 50
EMBEDDING_DIM = 32
MOOD_FEATURES = ["valence", "energy", "danceability", "acousticness", "tempo"]


def build_training_pairs(sessions, arcs, track_feats, user_emb, als_model, user_idx, item_idx):
    """
    For each session:
      - positive: tracks actually played later in session
      - negatives: ALS candidates NOT in session
    """
    track_feats_idx = track_feats.set_index("track")
    user_emb_idx    = user_emb.set_index("user")
    item_idx_rev    = {v: k for k, v in item_idx.items()}
    arc_map         = arcs.set_index("global_session_id")

    rows = []
    for sid, grp in sessions.groupby("global_session_id"):
        if sid not in arc_map.index:
            continue
        arc_row  = arc_map.loc[sid]
        user     = grp["user"].iloc[0]
        traj_cls = arc_row.get("trajectory_cluster", -1)
        hour     = arc_row.get("hour_of_day", 12)
        dow      = arc_row.get("day_of_week", 0)

        if user not in user_idx:
            continue

        uid = user_idx[user]
        # get user embedding
        u_emb = als_model.user_factors[uid]

        # positives: last half of session
        n = len(grp)
        positives = set(grp.iloc[n//2:]["track"].tolist())
        seed_tracks = set(grp.iloc[:n//2]["track"].tolist())

        # generate candidates via ALS
        try:
            cand_ids, _ = als_model.recommend(uid, als_model.user_factors[uid:uid+1],
                                               N=N_CANDIDATES, filter_already_liked_items=False)
            candidates = {item_idx_rev.get(c) for c in cand_ids if c in item_idx_rev}
        except Exception:
            candidates = set()

        candidates = (candidates | positives) - seed_tracks

        for track in candidates:
            if track not in track_feats_idx.index:
                continue
            tf = track_feats_idx.loc[track]
            label = 1 if track in positives else 0

            row = {
                "session_id":        sid,
                "track":             track,
                "label":             label,
                "trajectory_cluster": traj_cls,
                "hour_of_day":       hour,
                "day_of_week":       dow,
                "log_play_count":    tf.get("log_play_count", 0),
                "unique_users":      tf.get("unique_users", 0),
                "track_valence":     tf.get("mean_valence", 0),
                "track_energy":      tf.get("mean_energy", 0),
                "track_danceability":tf.get("mean_danceability", 0),
                "track_acousticness":tf.get("mean_acousticness", 0),
                "track_tempo":       tf.get("mean_tempo", 0),
            }
            # add user embedding dims
            for i, v in enumerate(u_emb):
                row[f"u_emb_{i}"] = v
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    sessions    = pd.read_csv(f"{DATA_DIR}/sessions.csv")
    arcs        = pd.read_csv(f"{DATA_DIR}/session_arcs_clustered.csv")
    track_feats = pd.read_parquet(f"{DATA_DIR}/track_features.parquet")
    user_emb    = pd.read_parquet(f"{DATA_DIR}/user_embeddings.parquet")
    als_model   = joblib.load(f"{MODELS_DIR}/als_model.pkl")
    user_idx    = joblib.load(f"{MODELS_DIR}/user_idx.pkl")
    item_idx    = joblib.load(f"{MODELS_DIR}/item_idx.pkl")

    print("Building training pairs...")
    df = build_training_pairs(sessions, arcs, track_feats, user_emb, als_model, user_idx, item_idx)
    print(f"Training pairs: {len(df):,} | Positive rate: {df['label'].mean():.3%}")

    feature_cols = [c for c in df.columns if c not in ["session_id", "track", "label"]]
    X = df[feature_cols].fillna(0)
    y = df["label"]
    groups = df["session_id"]

    # group-aware train/val split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val,   y_val   = X.iloc[val_idx],   y.iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val,   label=y_val, reference=train_data)

    params = {
        "objective":      "binary",
        "metric":         ["binary_logloss", "auc"],
        "num_leaves":     63,
        "learning_rate":  0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":   5,
        "min_child_samples": 20,
        "verbose":        -1,
    }

    mlflow.set_experiment("lightgbm-ranker")
    with mlflow.start_run():
        mlflow.log_params(params)

        print("Training LightGBM ranker...")
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(25)]
        model = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        best_iter = model.best_iteration
        best_auc  = model.best_score["val"]["auc"]
        mlflow.log_metrics({"best_iter": best_iter, "best_val_auc": best_auc})
        print(f"\nBest AUC: {best_auc:.4f} at iteration {best_iter}")

    joblib.dump({"model": model, "feature_cols": feature_cols}, f"{MODELS_DIR}/lgbm_ranker.pkl")
    print(f"Saved ranker → {MODELS_DIR}/lgbm_ranker.pkl")

    # feature importance
    imp = pd.Series(model.feature_importance(importance_type="gain"),
                    index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 features:")
    print(imp.head(10).to_string())


if __name__ == "__main__":
    main()
