"""
Step 2a: Segment scrobbles into listening sessions.
A new session starts when the gap between consecutive tracks > 30 minutes.
Also computes per-track mood vector and session-level mood arc.

Run: python session_segmenter.py
Input:  data/enriched_scrobbles.csv
Output: data/sessions.csv, data/session_arcs.csv
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "../data"
SESSION_GAP_SECONDS = 30 * 60  # 30 minutes
MIN_SESSION_LENGTH = 4          # drop sessions shorter than 4 tracks
MOOD_FEATURES = ["valence", "energy", "danceability", "acousticness", "tempo"]


def normalize_tempo(df):
    """Normalize tempo to 0-1 range."""
    df = df.copy()
    df["tempo"] = (df["tempo"] - df["tempo"].min()) / (df["tempo"].max() - df["tempo"].min() + 1e-8)
    return df


def assign_sessions(df):
    """Assign session IDs within each user's listening history."""
    df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)
    df["time_gap"] = df.groupby("user")["timestamp"].diff().fillna(0)
    df["new_session"] = (df["time_gap"] > SESSION_GAP_SECONDS) | (df["time_gap"] == 0)
    df["session_id"] = df.groupby("user")["new_session"].cumsum()
    df["global_session_id"] = df["user"] + "_" + df["session_id"].astype(str)
    return df


def compute_mood_arc(session_df):
    """
    Compute a mood arc for a session.
    Returns start, mid, and end mood vectors (valence + energy summary).
    This is the novel signal: how the session evolves emotionally.
    """
    n = len(session_df)
    third = max(1, n // 3)

    start = session_df.iloc[:third][MOOD_FEATURES].mean()
    mid   = session_df.iloc[third:2*third][MOOD_FEATURES].mean()
    end   = session_df.iloc[2*third:][MOOD_FEATURES].mean()

    arc = {}
    for f in MOOD_FEATURES:
        arc[f"start_{f}"] = start[f]
        arc[f"mid_{f}"]   = mid[f]
        arc[f"end_{f}"]   = end[f]
        arc[f"delta_{f}"] = end[f] - start[f]   # direction of change

    arc["session_length"] = n
    arc["valence_trend"]  = "up" if arc["delta_valence"] > 0.05 else ("down" if arc["delta_valence"] < -0.05 else "flat")
    arc["energy_trend"]   = "up" if arc["delta_energy"]  > 0.05 else ("down" if arc["delta_energy"]  < -0.05 else "flat")
    return arc


def main():
    df = pd.read_csv(f"{DATA_DIR}/enriched_scrobbles.csv")
    print(f"Loaded {len(df):,} enriched scrobbles")

    df = normalize_tempo(df)
    df = assign_sessions(df)

    # filter short sessions
    session_sizes = df.groupby("global_session_id").size()
    valid_sessions = session_sizes[session_sizes >= MIN_SESSION_LENGTH].index
    df = df[df["global_session_id"].isin(valid_sessions)]

    print(f"Sessions after filtering (min {MIN_SESSION_LENGTH} tracks): {df['global_session_id'].nunique():,}")

    # save track-level session data
    df.to_csv(f"{DATA_DIR}/sessions.csv", index=False)
    print(f"Saved sessions.csv")

    # compute session-level mood arcs
    print("Computing mood arcs...")
    arc_rows = []
    for sid, grp in df.groupby("global_session_id"):
        arc = compute_mood_arc(grp.reset_index(drop=True))
        arc["global_session_id"] = sid
        arc["user"] = grp["user"].iloc[0]
        arc["hour_of_day"] = pd.to_datetime(grp["timestamp"].iloc[0], unit="s").hour
        arc["day_of_week"]  = pd.to_datetime(grp["timestamp"].iloc[0], unit="s").dayofweek
        arc_rows.append(arc)

    arcs_df = pd.DataFrame(arc_rows)
    arcs_df.to_csv(f"{DATA_DIR}/session_arcs.csv", index=False)
    print(f"Saved session_arcs.csv with {len(arcs_df):,} sessions")
    print(arcs_df[["global_session_id", "session_length", "valence_trend", "energy_trend"]].head(10))


if __name__ == "__main__":
    main()
