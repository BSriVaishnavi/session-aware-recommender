"""
Step 1a: Collect listening history from Last.fm API.
Pulls recent tracks for a set of users and saves raw scrobbles.
Run: python lastfm_scraper.py
Output: data/raw_scrobbles.csv
"""

import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LASTFM_API_KEY")
BASE_URL = "http://ws.audioscrobbler.com/2.0/"
DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)


def get_top_users(n=200):
    """Pull active users from a public Last.fm group."""
    params = {
        "method": "group.getmembers",
        "group": "last-fm",
        "api_key": API_KEY,
        "format": "json",
        "limit": n,
    }
    r = requests.get(BASE_URL, params=params)
    data = r.json()
    if "members" not in data:
        # fallback: use a hardcoded seed list of known active public users
        return [
            "RJ", "Solbris", "bbc6music", "pitchfork", "allmusic",
            "thevinylfactory", "factmag", "xlrecordings", "warprecords",
            "ninja_tune", "subpop", "4ad_official", "mergerecords",
            "kranky", "ghostly", "erased_tapes", "morr_music",
            "temporary_residence", "polyvinyl", "secretly_canadian",
        ] * 10  # repeat to get ~200
    return [m["name"] for m in data["members"]["user"]]


def get_user_tracks(username, pages=5):
    """Fetch recent tracks for a user (200 tracks per page max)."""
    tracks = []
    for page in range(1, pages + 1):
        params = {
            "method": "user.getrecenttracks",
            "user": username,
            "api_key": API_KEY,
            "format": "json",
            "limit": 200,
            "page": page,
        }
        try:
            r = requests.get(BASE_URL, params=params, timeout=10)
            data = r.json()
            if "recenttracks" not in data:
                break
            track_list = data["recenttracks"].get("track", [])
            if not track_list:
                break
            for t in track_list:
                # skip currently playing track (no timestamp)
                if isinstance(t.get("date"), dict):
                    tracks.append({
                        "user": username,
                        "artist": t["artist"]["#text"],
                        "track": t["name"],
                        "album": t["album"]["#text"],
                        "timestamp": int(t["date"]["uts"]),
                    })
        except Exception as e:
            print(f"  Error fetching {username} page {page}: {e}")
            break
        time.sleep(0.25)  # respect rate limit
    return tracks


def main():
    print("Fetching user list...")
    users = get_top_users(200)
    users = list(set(users))[:150]  # dedupe, cap at 150

    all_tracks = []
    print(f"Fetching tracks for {len(users)} users...")
    for user in tqdm(users):
        tracks = get_user_tracks(user, pages=5)
        all_tracks.extend(tracks)
        time.sleep(0.1)

    df = pd.DataFrame(all_tracks)
    df = df.drop_duplicates()
    df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)

    out_path = f"{DATA_DIR}/raw_scrobbles.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} scrobbles from {df['user'].nunique()} users → {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
