"""
Step 3a: Mood Trajectory Clustering — the novel contribution.
Clusters session mood arcs into K archetypes using DTW + k-means.
These trajectory types become the conditioning signal for recommendations.

Run: python trajectory_clustering.py
Input:  data/session_arcs.csv
Output: data/session_arcs_clustered.csv, models/trajectory_model.pkl
        plots/trajectory_archetypes.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib
import os
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import StandardScaler

DATA_DIR  = "../data"
MODELS_DIR = "../models"
PLOTS_DIR  = "../plots"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

N_CLUSTERS    = 8
MOOD_FEATURES = ["valence", "energy", "danceability", "acousticness"]


def build_arc_sequences(arcs_df):
    """
    Build 3-timestep mood sequences [start, mid, end] per session.
    Shape: (n_sessions, 3, n_features)
    """
    sequences = []
    for _, row in arcs_df.iterrows():
        seq = []
        for phase in ["start", "mid", "end"]:
            seq.append([row[f"{phase}_{f}"] for f in MOOD_FEATURES])
        sequences.append(seq)
    return np.array(sequences, dtype=np.float32)


def cluster_trajectories(sequences, n_clusters=N_CLUSTERS):
    """Cluster mood arc sequences using DTW-based k-means."""
    scaler = TimeSeriesScalerMeanVariance()
    sequences_scaled = scaler.fit_transform(sequences)

    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        max_iter=50,
        random_state=42,
        n_jobs=-1,
    )
    labels = model.fit_predict(sequences_scaled)
    return model, scaler, labels


ARCHETYPE_NAMES = {
    0: "Chill Descent",      # high energy → low energy
    1: "Morning Lift",       # low → high valence/energy
    2: "Late Night Focus",   # low valence, stable energy
    3: "Party Arc",          # high danceability throughout
    4: "Emotional Journey",  # valence dips then recovers
    5: "Deep Work",          # low everything, stable
    6: "Sunset Wind-down",   # energy drops, valence stays warm
    7: "Euphoric Build",     # everything climbs
}


def plot_archetypes(model, scaler, feature_names=MOOD_FEATURES):
    """Plot the cluster centroids as mood arc trajectories."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    colors = ["#E07B54", "#5B8DB8", "#7DB87D", "#B87DB8",
              "#B8A55B", "#5BB8B8", "#B85B5B", "#8B5BB8"]

    for cluster_id in range(N_CLUSTERS):
        ax = axes[cluster_id]
        centroid = model.cluster_centers_[cluster_id]  # shape (3, n_features)

        for i, feat in enumerate(feature_names):
            ax.plot(["Start", "Mid", "End"], centroid[:, i],
                    marker="o", label=feat, linewidth=2)

        ax.set_title(f"#{cluster_id}: {ARCHETYPE_NAMES.get(cluster_id, f'Archetype {cluster_id}')}",
                     fontsize=10, fontweight="bold")
        ax.set_ylim(-2.5, 2.5)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=7)
        ax.set_facecolor("#f8f8f8")

    fig.suptitle("Mood Trajectory Archetypes (DTW K-Means Cluster Centroids)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/trajectory_archetypes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved archetype plot → {PLOTS_DIR}/trajectory_archetypes.png")


def main():
    arcs = pd.read_csv(f"{DATA_DIR}/session_arcs.csv")
    print(f"Loaded {len(arcs):,} session arcs")

    # drop any rows with missing arc features
    arc_cols = [f"{p}_{f}" for p in ["start", "mid", "end"] for f in MOOD_FEATURES]
    arcs = arcs.dropna(subset=arc_cols).reset_index(drop=True)
    print(f"Clean arcs: {len(arcs):,}")

    sequences = build_arc_sequences(arcs)
    print(f"Arc sequence tensor shape: {sequences.shape}")

    print(f"Clustering into {N_CLUSTERS} trajectory archetypes via DTW k-means...")
    model, scaler, labels = cluster_trajectories(sequences, N_CLUSTERS)

    arcs["trajectory_cluster"] = labels
    arcs["trajectory_name"]    = arcs["trajectory_cluster"].map(ARCHETYPE_NAMES)

    # cluster distribution
    dist = arcs["trajectory_name"].value_counts()
    print("\nCluster distribution:")
    print(dist.to_string())

    arcs.to_csv(f"{DATA_DIR}/session_arcs_clustered.csv", index=False)
    joblib.dump({"model": model, "scaler": scaler}, f"{MODELS_DIR}/trajectory_model.pkl")
    print(f"\nSaved clustered arcs → {DATA_DIR}/session_arcs_clustered.csv")
    print(f"Saved trajectory model → {MODELS_DIR}/trajectory_model.pkl")

    plot_archetypes(model, scaler)


if __name__ == "__main__":
    main()
