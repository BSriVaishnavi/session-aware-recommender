# Session-Aware Music Recommender with Mood Trajectory Modeling

A production-grade music recommendation system that models how listening sessions evolve emotionally over time — and recommends the next track based on *where the mood is heading*, not just what was played.

## What makes this different

Most recommender systems ask: *"what did users like?"*  
This system asks: *"where is this session's mood going, and what track fits that arc?"*

**Novel contributions:**
- **Mood Trajectory Clustering** — 8 session archetypes (e.g. "Chill Descent", "Euphoric Build") discovered via DTW k-means on Spotify audio feature sequences
- **Mood Coherence Score (MCS)** — a custom evaluation metric measuring how well recommendations fit the predicted mood trajectory (cosine similarity of audio features vs target mood vector)
- **GRU trajectory predictor** — predicts session archetype from the first 3 tracks, enabling trajectory-conditioned ranking
- **Hybrid ranking** — ALS collaborative filtering for candidate generation + LightGBM ranker conditioned on trajectory type

## Architecture

```
Last.fm API  ──► Session Segmentation ──► Mood Arc Computation
Spotify API  ──► Audio Feature Enrichment ──► DTW Clustering (8 archetypes)
                                                      │
                                               GRU Predictor
                                                      │
User seed tracks ──► Trajectory Prediction ──► ALS Candidates ──► LightGBM Ranker
                                                                         │
                                                              FastAPI + Redis Cache
                                                                         │
                                                              Streamlit "DJ Mode" UI
```

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
python lastfm_scraper.py       # ~20 min, pulls ~80K scrobbles
python spotify_enricher.py     # ~30 min, enriches with audio features

# Step 2: Preprocess
cd ../2_preprocessing
python session_segmenter.py    # segments into sessions, computes mood arcs
python feature_engineer.py     # ALS embeddings + feature engineering

# Step 3: Train models
cd ../3_modeling
python trajectory_clustering.py  # DTW k-means, ~5 min
python gru_trainer.py            # GRU trajectory predictor, ~10 min
python lightgbm_ranker.py        # LightGBM ranker, ~5 min

# Step 4: Evaluate
cd ../4_evaluation
python metrics.py               # NDCG, HitRate, MRR, MCS

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

## Evaluation metrics

| Metric | Description |
|--------|-------------|
| NDCG@K | Normalized Discounted Cumulative Gain — standard ranking quality |
| HitRate@K | Did any relevant track appear in top K? |
| MRR | Mean Reciprocal Rank |
| **MCS** | **Mood Coherence Score** — novel metric: cosine similarity of recommendations to predicted trajectory target mood |

## Tech stack

| Layer | Technology |
|-------|-----------|
| Data collection | Last.fm API, Spotify API (spotipy) |
| Mood modeling | DTW + k-means (tslearn) |
| Sequence model | PyTorch GRU |
| Ranker | LightGBM |
| Collaborative filtering | ALS (implicit) |
| Serving | FastAPI + Redis |
| Experiment tracking | MLflow |
| Frontend | Streamlit + Plotly |

## Resume bullets (fill in your actual numbers)

- Built a session-aware music recommender over **80K real Last.fm listening sessions**, modeling emotional mood trajectories using Spotify audio features (valence, energy, danceability)
- Designed a novel **Mood Coherence Score** evaluation metric measuring trajectory alignment of recommendations, achieving **0.74 cosine similarity** vs 0.51 for a popularity baseline
- Clustered **8 distinct session mood archetypes** using DTW + k-means on audio feature sequences, enabling arc-conditioned next-song prediction via a **PyTorch GRU**
- Deployed a real-time **"DJ Mode"** serving layer via **FastAPI + Redis**, generating mood-coherent playlists with **sub-100ms response time**

## Project structure

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
│   ├── lastfm_scraper.py
│   └── spotify_enricher.py
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
