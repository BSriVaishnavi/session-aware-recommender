"""
Step 5: FastAPI serving layer with Redis caching.
Exposes a /recommend endpoint that takes a seed track list
and returns mood-coherent recommendations in <100ms.

Run: uvicorn api:app --reload --port 8000
"""

import os
import json
import time
import hashlib
import numpy as np
import pandas as pd
import torch
import joblib
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "../data"
MODELS_DIR = "../models"
MOOD_FEATURES = ["valence", "energy", "danceability", "acousticness", "tempo"]
TOP_K = 10

app = FastAPI(
    title="Session-Aware Music Recommender",
    description="Mood trajectory-conditioned next-track recommendations",
    version="1.0.0",
)

# ── Load models at startup ────────────────────────────────────────────────────
print("Loading models...")

lgbm_bundle    = joblib.load(f"{MODELS_DIR}/lgbm_ranker.pkl")
lgbm_model     = lgbm_bundle["model"]
feature_cols   = lgbm_bundle["feature_cols"]

traj_bundle    = joblib.load(f"{MODELS_DIR}/trajectory_model.pkl")
traj_model     = traj_bundle["model"]
traj_scaler    = traj_bundle["scaler"]

gru_config     = joblib.load(f"{MODELS_DIR}/gru_config.pkl")
als_model      = joblib.load(f"{MODELS_DIR}/als_model.pkl")
user_idx       = joblib.load(f"{MODELS_DIR}/user_idx.pkl")
item_idx       = joblib.load(f"{MODELS_DIR}/item_idx.pkl")
item_idx_rev   = {v: k for k, v in item_idx.items()}

track_feats    = pd.read_parquet(f"{DATA_DIR}/track_features.parquet").set_index("track")

# Redis
try:
    cache = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
    )
    cache.ping()
    CACHE_ENABLED = True
    print("Redis connected.")
except Exception:
    CACHE_ENABLED = False
    print("Redis not available — running without cache.")

ARCHETYPE_NAMES = {
    0: "Chill Descent", 1: "Morning Lift", 2: "Late Night Focus",
    3: "Party Arc",     4: "Emotional Journey", 5: "Deep Work",
    6: "Sunset Wind-down", 7: "Euphoric Build",
}

ARCHETYPE_TARGET_MOODS = {
    0: np.array([0.3, 0.2, 0.3, 0.7, 0.3]),
    1: np.array([0.8, 0.8, 0.7, 0.2, 0.7]),
    2: np.array([0.3, 0.4, 0.3, 0.4, 0.4]),
    3: np.array([0.7, 0.9, 0.9, 0.1, 0.8]),
    4: np.array([0.6, 0.5, 0.5, 0.5, 0.5]),
    5: np.array([0.2, 0.3, 0.2, 0.6, 0.3]),
    6: np.array([0.6, 0.3, 0.4, 0.6, 0.3]),
    7: np.array([0.9, 0.9, 0.8, 0.1, 0.9]),
}

print("Models loaded. Ready.")


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    seed_tracks: List[str]           # list of track names already played
    user: Optional[str] = None       # optional: for personalized ALS candidates
    hour_of_day: Optional[int] = 12
    day_of_week: Optional[int] = 0
    top_k: Optional[int] = TOP_K


class TrackRecommendation(BaseModel):
    track: str
    score: float
    mood_coherence: float
    valence: float
    energy: float
    danceability: float


class RecommendResponse(BaseModel):
    trajectory_cluster: int
    trajectory_name: str
    recommendations: List[TrackRecommendation]
    latency_ms: float
    cached: bool


# ── Helper functions ──────────────────────────────────────────────────────────

def predict_trajectory(seed_tracks: List[str]) -> int:
    """Predict trajectory cluster from seed track audio features."""
    vecs = []
    for track in seed_tracks[:3]:
        if track in track_feats.index:
            row = track_feats.loc[track]
            vecs.append([row.get(f"mean_{f}", 0.5) for f in ["valence","energy","danceability","acousticness"]])
        else:
            vecs.append([0.5] * 4)
    while len(vecs) < 3:
        vecs.append([0.5] * 4)

    # flatten to 2D for StandardScaler
    flat = np.array(vecs).flatten().reshape(1, -1)
    flat_scaled = traj_scaler.transform(flat)
    label = traj_model.predict(flat_scaled)[0]
    return int(label)

def get_candidates(user: Optional[str], seed_tracks: List[str], n: int = 200) -> List[str]:
    """Get candidate tracks via ALS or fallback to popularity."""
    seed_set = set(seed_tracks)

    if user and user in user_idx:
        uid = user_idx[user]
        try:
            cand_ids, _ = als_model.recommend(
                uid, als_model.user_factors[uid:uid+1],
                N=n, filter_already_liked_items=False,
            )
            candidates = [item_idx_rev[c] for c in cand_ids if c in item_idx_rev]
            candidates = [t for t in candidates if t not in seed_set]
            if len(candidates) >= 20:
                return candidates[:n]
        except Exception:
            pass

    # fallback: popularity-based candidates
    pop = track_feats.nlargest(n * 2, "log_play_count").index.tolist()
    return [t for t in pop if t not in seed_set][:n]


def rank_candidates(candidates, traj_cls, hour, dow, top_k) -> List[dict]:
    """Score candidates with LightGBM ranker and compute MCS."""
    target_mood = ARCHETYPE_TARGET_MOODS.get(traj_cls, np.ones(5) * 0.5)
    rows = []

    for track in candidates:
        if track not in track_feats.index:
            continue
        tf = track_feats.loc[track]
        row = {
            "trajectory_cluster": traj_cls,
            "hour_of_day":        hour,
            "day_of_week":        dow,
            "log_play_count":     tf.get("log_play_count", 0),
            "unique_users":       tf.get("unique_users", 0),
            "track_valence":      tf.get("mean_valence", 0.5),
            "track_energy":       tf.get("mean_energy", 0.5),
            "track_danceability": tf.get("mean_danceability", 0.5),
            "track_acousticness": tf.get("mean_acousticness", 0.5),
            "track_tempo":        tf.get("mean_tempo", 0.5),
        }
        for i in range(32):
            row[f"u_emb_{i}"] = 0.0
        rows.append((track, row))

    if not rows:
        return []

    tracks_list = [r[0] for r in rows]
    X = pd.DataFrame([r[1] for r in rows]).reindex(columns=feature_cols, fill_value=0)
    scores = lgbm_model.predict(X)

    results = []
    for track, score in sorted(zip(tracks_list, scores), key=lambda x: -x[1])[:top_k]:
        tf = track_feats.loc[track]
        track_vec = np.array([tf.get(f"mean_{f}", 0.5) for f in MOOD_FEATURES])
        mcs = float(np.dot(track_vec, target_mood) /
                    (np.linalg.norm(track_vec) * np.linalg.norm(target_mood) + 1e-8))
        results.append({
            "track":           track,
            "score":           float(score),
            "mood_coherence":  mcs,
            "valence":         float(tf.get("mean_valence", 0)),
            "energy":          float(tf.get("mean_energy", 0)),
            "danceability":    float(tf.get("mean_danceability", 0)),
        })

    return results


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "cache": CACHE_ENABLED}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    t0 = time.time()

    if len(req.seed_tracks) < 1:
        raise HTTPException(status_code=400, detail="Provide at least 1 seed track.")

    # cache key
    cache_key = hashlib.md5(
        json.dumps({"tracks": sorted(req.seed_tracks), "user": req.user,
                    "hour": req.hour_of_day, "dow": req.day_of_week}).encode()
    ).hexdigest()

    if CACHE_ENABLED:
        cached = cache.get(f"rec:{cache_key}")
        if cached:
            data = json.loads(cached)
            data["cached"] = True
            data["latency_ms"] = round((time.time() - t0) * 1000, 2)
            return RecommendResponse(**data)

    traj_cls   = predict_trajectory(req.seed_tracks)
    candidates = get_candidates(req.user, req.seed_tracks)
    recs       = rank_candidates(candidates, traj_cls, req.hour_of_day, req.day_of_week, req.top_k)

    response_data = {
        "trajectory_cluster": traj_cls,
        "trajectory_name":    ARCHETYPE_NAMES.get(traj_cls, f"Archetype {traj_cls}"),
        "recommendations":    recs,
        "latency_ms":         round((time.time() - t0) * 1000, 2),
        "cached":             False,
    }

    if CACHE_ENABLED:
        cache.setex(f"rec:{cache_key}", 3600, json.dumps(response_data))

    return RecommendResponse(**response_data)


@app.get("/trajectory/{cluster_id}")
def get_trajectory_info(cluster_id: int):
    """Return info about a trajectory archetype."""
    if cluster_id not in ARCHETYPE_NAMES:
        raise HTTPException(status_code=404, detail="Unknown cluster ID")
    return {
        "cluster_id":   cluster_id,
        "name":         ARCHETYPE_NAMES[cluster_id],
        "target_mood":  {f: v for f, v in zip(MOOD_FEATURES, ARCHETYPE_TARGET_MOODS[cluster_id])},
    }
