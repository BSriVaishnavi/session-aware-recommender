"""
Step 1 (rewritten): Load data from Kaggle files instead of APIs.
Reads the Last.fm TSV and Spotify CSV directly — no API keys needed.

Run: python load_kaggle_data.py
Input:  data/userid-timestamp-artid-artname-traid-traname.tsv
        data/SpotifyAudioFeaturesNov2018.csv
Output: data/enriched_scrobbles.csv
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

DATA_DIR = "data"

AUDIO_FEATURES = [
    "valence", "energy", "danceability", "tempo",
    "acousticness", "instrumentalness", "liveness",
    "loudness", "speechiness", "mode",
]


def load_scrobbles(n_users=500):
    """Load listening history from Last.fm TSV, keep first N users."""
    print("Loading Last.fm scrobbles (this takes ~1 min for large file)...")
    chunks = []
    chunk_size = 500_000

    reader = pd.read_csv(
        f"{DATA_DIR}/userid-timestamp-artid-artname-traid-traname.tsv",
        sep="\t",
        header=None,
        names=["user", "timestamp", "artist_id", "artist", "track_id", "track"],
        usecols=["user", "timestamp", "artist", "track"],
        encoding="utf-8",
        on_bad_lines="skip",
        chunksize=chunk_size,
    )

    users_seen = set()
    for chunk in reader:
        chunk = chunk.dropna(subset=["track"])
        users_seen.update(chunk["user"].unique())
        chunks.append(chunk)
        if len(users_seen) >= n_users:
            break

    df = pd.concat(chunks, ignore_index=True)

    # keep only first n_users
    top_users = list(df["user"].unique())[:n_users]
    df = df[df["user"].isin(top_users)]

    # convert timestamp to unix
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].astype(np.int64) // 10**9

    df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)
    print(f"Loaded {len(df):,} scrobbles from {df['user'].nunique()} users")
    return df


def load_spotify_features():
    """Load pre-collected Spotify audio features from Kaggle CSV."""
    print("Loading Spotify audio features...")
    sp = pd.read_csv(f"{DATA_DIR}/SpotifyAudioFeaturesNov2018.csv")
    print(f"Spotify dataset columns: {list(sp.columns)}")

    # normalize column names
    sp.columns = sp.columns.str.strip().str.lower().str.replace(" ", "_")

    # find track name column
    name_col = None
    for c in ["track_name", "name", "song_name", "title"]:
        if c in sp.columns:
            name_col = c
            break

    if name_col is None:
        print("Could not find track name column. Columns:", list(sp.columns))
        raise ValueError("No track name column found in Spotify CSV")

    sp = sp.rename(columns={name_col: "track"})
    sp["track"] = sp["track"].str.strip().str.lower()

    # keep only what we need
    keep = ["track"] + [f for f in AUDIO_FEATURES if f in sp.columns]
    sp = sp[keep].drop_duplicates(subset=["track"])
    print(f"Spotify features: {len(sp):,} unique tracks")
    return sp


def merge_and_enrich(scrobbles, spotify):
    """Merge scrobbles with Spotify audio features on track name."""
    print("Merging on track name...")
    scrobbles["track_lower"] = scrobbles["track"].str.strip().str.lower()
    spotify["track_lower"]   = spotify["track"].str.strip().str.lower()

    enriched = scrobbles.merge(
        spotify.drop(columns=["track"]),
        on="track_lower",
        how="left",
    ).drop(columns=["track_lower"])

    matched = enriched["valence"].notna().sum()
    total   = len(enriched)
    print(f"Matched {matched:,} / {total:,} scrobbles with audio features ({matched/total:.1%})")

    # keep only matched rows
    enriched = enriched.dropna(subset=["valence"]).reset_index(drop=True)
    print(f"Final enriched dataset: {len(enriched):,} scrobbles")
    return enriched


def normalize_tempo(df):
    df = df.copy()
    df["tempo"] = (df["tempo"] - df["tempo"].min()) / (df["tempo"].max() - df["tempo"].min() + 1e-8)
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    scrobbles = load_scrobbles(n_users=500)
    spotify   = load_spotify_features()
    enriched  = merge_and_enrich(scrobbles, spotify)
    enriched  = normalize_tempo(enriched)

    out_path = f"{DATA_DIR}/enriched_scrobbles.csv"
    enriched.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(enriched[["user", "artist", "track", "valence", "energy", "timestamp"]].head(10))


if __name__ == "__main__":
    main()
