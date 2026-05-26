"""
Step 6: Streamlit "DJ Mode" frontend.
Enter seed tracks → see predicted mood trajectory → get recommendations.

Run: streamlit run app.py
"""

import streamlit as st
import time
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

API_URL = "http://localhost:8000"

MOOD_FEATURES = ["valence", "energy", "danceability", "acousticness", "tempo"]

ARCHETYPE_COLORS = {
    "Chill Descent":      "#5B8DB8",
    "Morning Lift":       "#F5A623",
    "Late Night Focus":   "#7B68EE",
    "Party Arc":          "#E07B54",
    "Emotional Journey":  "#B87DB8",
    "Deep Work":          "#5BB8B8",
    "Sunset Wind-down":   "#E08080",
    "Euphoric Build":     "#7DB87D",
}

st.set_page_config(
    page_title="🎵 DJ Mode — Mood-Aware Recommender",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 DJ Mode")
st.markdown("*Session-aware music recommendations powered by mood trajectory modeling*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Session Context")
    user = st.text_input("Username (optional)", placeholder="e.g. RJ")
    hour = st.slider("Hour of day", 0, 23, 20)
    dow  = st.selectbox("Day of week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    dow_idx = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(dow)
    top_k = st.slider("Recommendations", 5, 20, 10)
    st.divider()
    st.caption("Built with Last.fm + Spotify APIs")
    st.caption("Model: GRU + LightGBM + ALS")

# ── Main: Seed Track Input ────────────────────────────────────────────────────
st.subheader("🎧 What have you been listening to?")
st.caption("Enter 3–5 tracks you've already played this session")

col1, col2, col3 = st.columns(3)
with col1:
    track1 = st.text_input("Track 1", placeholder="e.g. Midnight City")
with col2:
    track2 = st.text_input("Track 2", placeholder="e.g. Do I Wanna Know?")
with col3:
    track3 = st.text_input("Track 3", placeholder="e.g. Redbone")

col4, col5 = st.columns(2)
with col4:
    track4 = st.text_input("Track 4 (optional)", placeholder="e.g. Electric Feel")
with col5:
    track5 = st.text_input("Track 5 (optional)", placeholder="e.g. Tame Impala")

seed_tracks = [t for t in [track1, track2, track3, track4, track5] if t.strip()]




st.info("💡 **P.S.** Mood trajectories may vary for common song names (e.g. 'Intro', 'Home') since multiple artists share the same title. For consistent results, use the suggested tracks below.")

# ── Song Suggestions ──────────────────────────────────────────────────
st.divider()
st.subheader("🎵 Not sure what to type? Try these:")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Chill Descent**")
    st.caption("Heartless · Angel · Breathe")
    st.markdown("**Late Night Focus**")
    st.caption("All I Need · Wonderwall · Run")
with col2:
    st.markdown("**Sunset Wind-down**")
    st.caption("Love Lockdown · Closer · Hurt")
    st.markdown("**Euphoric Build**")
    st.caption("Crazy · Street Lights · Runaway")
with col3:
    st.markdown("**Party Arc**")
    st.caption("Heartbeat · Get Back · Star")
    st.markdown("**Emotional Journey**")
    st.caption("I Want You · Hunter · Lost")

recommend_btn = st.button("🎵 Generate Playlist", type="primary", use_container_width=True)

# ── API Call + Results ────────────────────────────────────────────────────────
if recommend_btn:
    if len(seed_tracks) < 1:
        st.warning("Please enter at least 1 seed track.")
    else:
        with st.spinner("🎵 Predicting your mood arc and generating recommendations..."):
            try:
                time.sleep(1.5)
                resp = requests.post(f"{API_URL}/recommend", json={
                    "seed_tracks": seed_tracks,
                    "user":        user if user else None,
                    "hour_of_day": hour,
                    "day_of_week": dow_idx,
                    "top_k":       top_k,
                }, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Make sure `uvicorn api:app --reload --port 8000` is running in 5_serving/")
                st.stop()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        traj_name  = data["trajectory_name"]
        traj_color = ARCHETYPE_COLORS.get(traj_name, "#888888")
        recs       = data["recommendations"]
        latency    = data["latency_ms"]
        cached     = data["cached"]

        # ── Trajectory Banner ────────────────────────────────────────────────
        st.divider()
        st.subheader("🌊 Your Session's Mood Trajectory")

        col_traj, col_meta = st.columns([2, 1])
        with col_traj:
            st.markdown(
                f"""<div style="background:{traj_color}22;border-left:4px solid {traj_color};
                padding:16px;border-radius:8px;margin-bottom:8px">
                <h3 style="margin:0;color:{traj_color}">#{data['trajectory_cluster']}: {traj_name}</h3>
                <p style="margin:4px 0 0;color:#666;font-size:14px">
                Detected from your listening arc — recommendations are conditioned on this trajectory
                </p></div>""",
                unsafe_allow_html=True,
            )
        with col_meta:
            st.metric("Response time", f"{latency:.1f} ms")
            st.metric("Source", "Cache ⚡" if cached else "Live 🔴")
            st.metric("Tracks analyzed", len(seed_tracks))

        # ── Mood Arc Visualization ───────────────────────────────────────────
        st.subheader("📊 Mood Coherence of Recommendations")

        recs_df = pd.DataFrame(recs)
        if not recs_df.empty:
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=recs_df["track"],
                y=recs_df["mood_coherence"],
                marker_color=traj_color,
                name="Mood Coherence Score",
                opacity=0.85,
            ))
            fig.add_trace(go.Scatter(
                x=recs_df["track"],
                y=recs_df["valence"],
                mode="lines+markers",
                name="Valence",
                line=dict(color="#F5A623", width=2),
                yaxis="y",
            ))
            fig.add_trace(go.Scatter(
                x=recs_df["track"],
                y=recs_df["energy"],
                mode="lines+markers",
                name="Energy",
                line=dict(color="#E07B54", width=2),
                yaxis="y",
            ))

            fig.update_layout(
                xaxis_tickangle=-35,
                yaxis=dict(range=[0, 1], title="Score"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=380,
                margin=dict(l=20, r=20, t=20, b=80),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Recommendation Table ─────────────────────────────────────────────
        st.subheader(f"🎶 Your {top_k}-Track Playlist")

        for i, rec in enumerate(recs, 1):
            mcs = rec["mood_coherence"]
            bar_width = int(mcs * 100)
            bar_color = traj_color

            with st.container():
                c1, c2, c3, c4, c5 = st.columns([0.5, 3, 1.5, 1.5, 2])
                with c1:
                    st.markdown(f"**{i}**")
                with c2:
                    st.markdown(f"**{rec['track']}**")
                with c3:
                    st.markdown(f"⚡ Energy: `{rec['energy']:.2f}`")
                with c4:
                    st.markdown(f"😊 Valence: `{rec['valence']:.2f}`")
                with c5:
                    st.markdown(
                        f"""<div style="background:#eee;border-radius:4px;height:8px;margin-top:8px">
                        <div style="background:{bar_color};width:{bar_width}%;height:100%;border-radius:4px"></div>
                        </div><small style="color:#888">MCS: {mcs:.2f}</small>""",
                        unsafe_allow_html=True,
                    )

        # ── Scatter: Valence vs Energy ───────────────────────────────────────
        if not recs_df.empty:
            st.subheader("🎯 Recommendation Map — Valence vs Energy")
            fig2 = px.scatter(
                recs_df, x="valence", y="energy",
                size="mood_coherence", color="mood_coherence",
                hover_name="track",
                color_continuous_scale="Oranges",
                labels={"valence": "Valence (happiness →)", "energy": "Energy →"},
                range_x=[0, 1], range_y=[0, 1],
            )
            fig2.update_layout(
                height=380,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig2, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Session-Aware Music Recommender · Last.fm + Spotify APIs · GRU + LightGBM + ALS · MLflow")
