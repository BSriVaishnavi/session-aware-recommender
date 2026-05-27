# Session-Aware Music Recommender with Mood Trajectory Modeling

A production-grade music recommendation system that models how listening sessions evolve emotionally over time — recommending the next track based on *where the mood is heading*, not just what was played.

**Live Demo: https://bsrivaishnavi-session-aware-recommender.hf.space**

---

## What makes this different

Most recommender systems ask: *"what did users like?"*  
This system asks: *"where is this session's mood going, and what track fits that arc?"*

**Novel contributions:**
- **Mood Trajectory Clustering** — 8 session archetypes (e.g. "Chill Descent", "Euphoric Build") discovered via MiniBatchKMeans on Spotify audio feature sequences
- **Mood Coherence Score (MCS)** — a custom evaluation metric measuring how well recommendations fit the predicted mood trajectory (cosine similarity of audio features vs target mood vector)
- **GRU trajectory predictor** — predicts session archetype from the first 3 tracks, enabling trajectory-conditioned ranking (49.9% accuracy, 4× random baseline)
- **Hybrid ranking** — ALS collaborative filtering for candidate generation + LightGBM ranker (AUC 0.90) conditioned on trajectory type

---

## Architecture

```
Last.fm API  ──► Session Segmentation ──► Mood Arc Computation
Spotify API  ──► Audio Feature Enrichment ──► Clustering (8 archetypes)
                                                      │
                                               GRU Predictor
                                                      │
User seed tracks ──► Trajectory Prediction ──► ALS Candidates ──► LightGBM Ranker
                                                                         │
                                                              FastAPI + Redis Cache
                                                                         │
                                                              Streamlit "DJ Mode" UI
```

---

## Results

| Metric | Value |
|--------|-------|
| Training scrobbles | 1.87M (499 users) |
| Sessions | 164K |
| Trajectory archetypes | 8 |
| GRU trajectory accuracy | 49.9% (4× random baseline of 12.5%) |
| LightGBM ranker AUC | 0.90 on 4.2M session-track pairs |
| Mood Coherence Score | 0.864 vs 0.850 popularity baseline |
| API response time | ~25ms |
| NDCG@10 | 0.031 |
| Hit Rate@10 | 0.079 |

---

## Try These Songs

Use 3 tracks together for best trajectory prediction:

| Archetype | Track 1 | Track 2 | Track 3 |
|-----------|---------|---------|---------|
| Chill Descent | Heartless | Angel | Breathe |
| Late Night Focus | All I Need | Wonderwall | Run |
| Party Arc | Heartbeat | Get Back | Star |
| Emotional Journey | I Want You | Hunter | Lost |
| Sunset Wind-down | Love Lockdown | Closer | Hurt |
| Euphoric Build | Crazy | Street Lights | Home |

> **Tip:** Order matters — the same 3 songs in different order can produce different trajectories. Common song names (e.g. "Intro", "Home") may match different artists and produce varied results.

---

## Data Collection Pipeline

Built a fully dynamic data collection pipeline — no static datasets:

- **Last.fm social graph traversal** — discovers 200+ active users starting from a seed user via `user.getFriends` API, filtering for users with 1000+ scrobbles
- **Live scrobble collection** — pulls real listening history via `user.getrecenttracks` API with checkpoint saving
- **Audio feature enrichment** — enriched with Spotify audio features (valence, energy, danceability, tempo, acousticness)

> **Note on Spotify API:** Spotify deprecated their Audio Features endpoint for new Developer Mode apps in February 2026 as part of platform security updates. Production models use 1.87M historical scrobbles enriched with pre-collected Spotify audio features for optimal coverage. The dynamic collection pipeline is available in `1_data_collection/` for future use.

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up API keys
```bash
cp .env.example .env
# Edit .env and fill in your Last.fm and Spotify API keys
```

Get free API keys:
- Last.fm: https://www.last.fm/api/account/create
- Spotify: https://developer.spotify.com/dashboard

### 3. Run the pipeline in order

```bash
# Step 1: Collect data
cd 1_data_collection
python lastfm_scraper.py          # pulls scrobbles via Last.fm API
python spotify_enricher.py        # enriches with audio features

# Step 2: Preprocess
cd ../2_preprocessing
python session_segmenter.py       # segments into sessions, computes mood arcs
python feature_engineer.py        # ALS embeddings + feature engineering

# Step 3: Train models
cd ../3_modeling
python trajectory_clustering.py   # MiniBatchKMeans mood archetypes
python gru_trainer.py             # GRU trajectory predictor
python lightgbm_ranker.py         # LightGBM ranker

# Step 4: Evaluate
cd ../4_evaluation
python metrics.py                 # NDCG, HitRate, MRR, MCS

# Step 5: Serve
cd ../5_serving
uvicorn api:app --reload --port 8000

# Step 6: Frontend (new terminal)
cd ../6_frontend
streamlit run app.py
```

### 4. Optional: Install Redis for caching
```bash
# Windows: download from https://redis.io/download
# Then run: redis-server
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| NDCG@K | Normalized Discounted Cumulative Gain — standard ranking quality |
| HitRate@K | Did any relevant track appear in top K? |
| MRR | Mean Reciprocal Rank |
| **MCS** | **Mood Coherence Score** — novel metric: cosine similarity of recommendations to predicted trajectory target mood |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data collection | Last.fm API, Spotify API (spotipy) |
| Mood modeling | MiniBatchKMeans |
| Sequence model | PyTorch GRU |
| Ranker | LightGBM |
| Collaborative filtering | ALS (implicit) |
| Serving | FastAPI + Redis |
| Experiment tracking | MLflow |
| Frontend | Streamlit + Plotly |
| Deployment | HuggingFace Docker |

---

## Project Structure

```
music-recommender/
├── .env.example
├── requirements.txt
├── README.md
├── data/                        # generated after running pipeline
├── models/                      # saved model artifacts
├── plots/                       # evaluation plots
├── results/                     # evaluation CSVs
├── 1_data_collection/
│   ├── lastfm_scraper.py        # dynamic user discovery via Last.fm social graph
│   └── spotify_enricher.py      # audio feature enrichment via Spotify API
├── 2_preprocessing/
│   ├── session_segmenter.py
│   └── feature_engineer.py
├── 3_modeling/
│   ├── trajectory_clustering.py
│   ├── gru_trainer.py
│   └── lightgbm_ranker.py
├── 4_evaluation/
│   └── metrics.py
├── 5_serving/
│   └── api.py
└── 6_frontend/
    └── app.py
```
