"""
Step 4: Evaluation metrics.
Computes standard ranking metrics + the novel Mood Coherence Score.

Mood Coherence Score (MCS):
  Measures how well recommended tracks fit the predicted mood trajectory.
  = mean cosine similarity between recommended track's audio features
    and the target mood vector of the predicted trajectory archetype.

This is the novel evaluation contribution — analogous to Cohen's Kappa
in the Florida NER project.

Run: python metrics.py
Input:  data/sessions.csv, data/session_arcs_clustered.csv,
        models/lgbm_ranker.pkl, models/trajectory_model.pkl, models/gru_model.pt
Output: results/evaluation_report.csv, plots/metrics_comparison.png
"""

import pandas as pd
import numpy as np
import torch
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR    = "../data"
MODELS_DIR  = "../models"
RESULTS_DIR = "../results"
PLOTS_DIR   = "../plots"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MOOD_FEATURES = ["valence", "energy", "danceability", "acousticness", "tempo"]
K_VALUES      = [5, 10, 20]


# ── Standard metrics ──────────────────────────────────────────────────────────

def dcg_at_k(relevance, k):
    relevance = np.array(relevance[:k])
    if len(relevance) == 0:
        return 0.0
    gains = relevance / np.log2(np.arange(2, len(relevance) + 2))
    return gains.sum()


def ndcg_at_k(recommended, relevant_set, k):
    relevance = [1 if t in relevant_set else 0 for t in recommended[:k]]
    ideal     = sorted(relevance, reverse=True)
    dcg  = dcg_at_k(relevance, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended, relevant_set, k):
    return int(any(t in relevant_set for t in recommended[:k]))


def mean_reciprocal_rank(recommended, relevant_set):
    for i, t in enumerate(recommended):
        if t in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


# ── Novel metric: Mood Coherence Score ────────────────────────────────────────

def compute_mood_coherence_score(recommended_tracks, track_feats_df, target_mood_vector):
    """
    MCS = mean cosine similarity between each recommended track's
    audio features and the session's target mood vector.

    target_mood_vector: array of shape (n_features,) representing
    the predicted end-state of the session's mood trajectory.
    """
    tf_idx = track_feats_df.set_index("track")
    vectors = []
    for track in recommended_tracks:
        if track in tf_idx.index:
            row = tf_idx.loc[track]
            vec = np.array([row.get(f"mean_{f}", 0) for f in MOOD_FEATURES])
            vectors.append(vec)

    if not vectors:
        return 0.0

    target = target_mood_vector.reshape(1, -1)
    sims   = cosine_similarity(np.array(vectors), target)
    return float(sims.mean())


def popularity_baseline(track_feats, k):
    """Return top-k most popular tracks as the baseline recommendation."""
    return track_feats.nlargest(k, "log_play_count")["track"].tolist()


# ── Target mood vectors per trajectory archetype ─────────────────────────────

ARCHETYPE_TARGET_MOODS = {
    # Each archetype's target end-state mood [valence, energy, danceability, acousticness, tempo_normalized]
    0: np.array([0.3, 0.2, 0.3, 0.7, 0.3]),   # Chill Descent
    1: np.array([0.8, 0.8, 0.7, 0.2, 0.7]),   # Morning Lift
    2: np.array([0.3, 0.4, 0.3, 0.4, 0.4]),   # Late Night Focus
    3: np.array([0.7, 0.9, 0.9, 0.1, 0.8]),   # Party Arc
    4: np.array([0.6, 0.5, 0.5, 0.5, 0.5]),   # Emotional Journey
    5: np.array([0.2, 0.3, 0.2, 0.6, 0.3]),   # Deep Work
    6: np.array([0.6, 0.3, 0.4, 0.6, 0.3]),   # Sunset Wind-down
    7: np.array([0.9, 0.9, 0.8, 0.1, 0.9]),   # Euphoric Build
}


def evaluate(sessions, arcs, track_feats, lgbm_bundle, n_eval=500):
    model        = lgbm_bundle["model"]
    feature_cols = lgbm_bundle["feature_cols"]
    arc_map      = arcs.set_index("global_session_id")

    results = {k: {"ndcg": [], "hit_rate": [], "mrr": [], "mcs_model": [], "mcs_baseline": []}
               for k in K_VALUES}

    eval_sessions = list(sessions["global_session_id"].unique())[:n_eval]
    track_feats_idx = track_feats.set_index("track")

    for sid in eval_sessions:
        if sid not in arc_map.index:
            continue
        arc_row  = arc_map.loc[sid]
        traj_cls = int(arc_row.get("trajectory_cluster", 0))
        grp      = sessions[sessions["global_session_id"] == sid].sort_values("timestamp")

        n = len(grp)
        if n < 6:
            continue

        seed_tracks     = grp.iloc[:n//2]["track"].tolist()
        relevant_tracks = set(grp.iloc[n//2:]["track"].tolist())
        target_mood     = ARCHETYPE_TARGET_MOODS.get(traj_cls, np.ones(5) * 0.5)

        # build candidate features for all tracks not in seed
        candidates = track_feats[~track_feats["track"].isin(seed_tracks)].head(200)
        if len(candidates) == 0:
            continue

        feat_rows = []
        for _, crow in candidates.iterrows():
            row = {
                "trajectory_cluster": traj_cls,
                "hour_of_day":        arc_row.get("hour_of_day", 12),
                "day_of_week":        arc_row.get("day_of_week", 0),
                "log_play_count":     crow.get("log_play_count", 0),
                "unique_users":       crow.get("unique_users", 0),
                "track_valence":      crow.get("mean_valence", 0),
                "track_energy":       crow.get("mean_energy", 0),
                "track_danceability": crow.get("mean_danceability", 0),
                "track_acousticness": crow.get("mean_acousticness", 0),
                "track_tempo":        crow.get("mean_tempo", 0),
            }
            for i in range(32):
                row[f"u_emb_{i}"] = 0.0
            feat_rows.append(row)

        X_cand = pd.DataFrame(feat_rows).reindex(columns=feature_cols, fill_value=0)
        scores = model.predict(X_cand)

        ranked_tracks     = candidates["track"].tolist()
        ranked_by_score   = [t for _, t in sorted(zip(scores, ranked_tracks), reverse=True)]
        baseline_ranked   = popularity_baseline(track_feats, max(K_VALUES))

        for k in K_VALUES:
            results[k]["ndcg"].append(ndcg_at_k(ranked_by_score, relevant_tracks, k))
            results[k]["hit_rate"].append(hit_rate_at_k(ranked_by_score, relevant_tracks, k))
            results[k]["mrr"].append(mean_reciprocal_rank(ranked_by_score, relevant_tracks))
            results[k]["mcs_model"].append(compute_mood_coherence_score(ranked_by_score[:k], track_feats, target_mood))
            results[k]["mcs_baseline"].append(compute_mood_coherence_score(baseline_ranked[:k], track_feats, target_mood))

    return results


def print_and_save_results(results):
    rows = []
    for k, metrics in results.items():
        row = {
            "K":             k,
            "NDCG@K":        np.mean(metrics["ndcg"]),
            "HitRate@K":     np.mean(metrics["hit_rate"]),
            "MRR":           np.mean(metrics["mrr"]),
            "MCS_model":     np.mean(metrics["mcs_model"]),
            "MCS_baseline":  np.mean(metrics["mcs_baseline"]),
            "MCS_improvement": (np.mean(metrics["mcs_model"]) - np.mean(metrics["mcs_baseline"])) / (np.mean(metrics["mcs_baseline"]) + 1e-8),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n=== Evaluation Results ===")
    print(df.to_string(index=False, float_format="%.4f"))
    df.to_csv(f"{RESULTS_DIR}/evaluation_report.csv", index=False)
    print(f"\nSaved → {RESULTS_DIR}/evaluation_report.csv")
    return df


def plot_results(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(df["K"].astype(str), df["NDCG@K"], color="#5B8DB8")
    axes[0].set_title("NDCG@K"); axes[0].set_xlabel("K"); axes[0].set_ylim(0, 1)

    axes[1].bar(df["K"].astype(str), df["HitRate@K"], color="#7DB87D")
    axes[1].set_title("Hit Rate@K"); axes[1].set_xlabel("K"); axes[1].set_ylim(0, 1)

    x = np.arange(len(df))
    w = 0.35
    axes[2].bar(x - w/2, df["MCS_model"],    width=w, label="Our model", color="#E07B54")
    axes[2].bar(x + w/2, df["MCS_baseline"], width=w, label="Popularity baseline", color="#B4B2A9")
    axes[2].set_xticks(x); axes[2].set_xticklabels(df["K"].astype(str))
    axes[2].set_title("Mood Coherence Score"); axes[2].set_xlabel("K"); axes[2].set_ylim(0, 1)
    axes[2].legend()

    plt.suptitle("Session-Aware Music Recommender — Evaluation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot → {PLOTS_DIR}/metrics_comparison.png")


def main():
    sessions    = pd.read_csv(f"{DATA_DIR}/sessions.csv")
    arcs        = pd.read_csv(f"{DATA_DIR}/session_arcs_clustered.csv")
    track_feats = pd.read_parquet(f"{DATA_DIR}/track_features.parquet")
    lgbm_bundle = joblib.load(f"{MODELS_DIR}/lgbm_ranker.pkl")

    print("Running evaluation on 500 sessions...")
    results = evaluate(sessions, arcs, track_feats, lgbm_bundle, n_eval=500)
    df = print_and_save_results(results)
    plot_results(df)


if __name__ == "__main__":
    main()
