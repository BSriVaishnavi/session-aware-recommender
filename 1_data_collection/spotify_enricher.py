"""
Step 1b: Enrich scrobbles with Spotify audio features.
For each unique (artist, track) pair, fetches:
  valence, energy, danceability, tempo, acousticness,
  instrumentalness, liveness, loudness, speechiness, mode

Run: python spotify_enricher.py
Input:  data/raw_scrobbles.csv
Output: data/enriched_scrobbles.csv
"""

import os
import time
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
))

DATA_DIR = "../data"
AUDIO_FEATURES = [
    "valence", "energy", "danceability", "tempo",
    "acousticness", "instrumentalness", "liveness",
    "loudness", "speechiness", "mode",
]


def search_track_id(artist, track):
    """Search Spotify for a track and return its ID."""
    try:
        query = f"track:{track} artist:{artist}"
        results = sp.search(q=query, type="track", limit=1)
        items = results["tracks"]["items"]
        if items:
            return items[0]["id"]
    except Exception:
        pass
    return None


def get_audio_features_batch(track_ids):
    """Fetch audio features for up to 100 track IDs at once."""
    try:
        features = sp.audio_features(track_ids)
        return features
    except Exception:
        return [None] * len(track_ids)


def main():
    df = pd.read_csv(f"{DATA_DIR}/raw_scrobbles.csv")
    print(f"Loaded {len(df):,} scrobbles")

    # get unique tracks
    unique_tracks = df[["artist", "track"]].drop_duplicates().reset_index(drop=True)
    print(f"Unique (artist, track) pairs: {len(unique_tracks):,}")

    # search for Spotify IDs
    print("\nSearching Spotify track IDs...")
    ids = []
    for _, row in tqdm(unique_tracks.iterrows(), total=len(unique_tracks)):
        tid = search_track_id(row["artist"], row["track"])
        ids.append(tid)
        time.sleep(0.05)
    unique_tracks["spotify_id"] = ids

    # fetch audio features in batches of 100
    print("\nFetching audio features...")
    feature_rows = []
    valid = unique_tracks.dropna(subset=["spotify_id"])
    batch_size = 100

    for i in tqdm(range(0, len(valid), batch_size)):
        batch = valid.iloc[i:i+batch_size]
        feats = get_audio_features_batch(batch["spotify_id"].tolist())
        for row, feat in zip(batch.itertuples(), feats):
            if feat:
                feature_rows.append({
                    "artist": row.artist,
                    "track": row.track,
                    "spotify_id": row.spotify_id,
                    **{f: feat.get(f) for f in AUDIO_FEATURES},
                })
        time.sleep(0.1)

    features_df = pd.DataFrame(feature_rows)
    print(f"\nGot audio features for {len(features_df):,} / {len(unique_tracks):,} unique tracks")

    # merge back
    enriched = df.merge(features_df, on=["artist", "track"], how="left")
    enriched = enriched.dropna(subset=["valence"])  # drop tracks with no features

    out_path = f"{DATA_DIR}/enriched_scrobbles.csv"
    enriched.to_csv(out_path, index=False)
    print(f"Saved {len(enriched):,} enriched scrobbles → {out_path}")
    print(enriched[["user", "artist", "track", "valence", "energy", "timestamp"]].head())


if __name__ == "__main__":
    main()
