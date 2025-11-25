# movie_recommendation_app_fixed.py
# Full single-file Streamlit app — corrected & hardened version of the user's code.
# Save as movie_recommendation_app_fixed.py and run:
#    streamlit run movie_recommendation_app_fixed.py

import os
import base64
import pickle
import requests
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Tuple, List
from textwrap import dedent

# --------------------------
# Config / Constants
# --------------------------
BG_IMAGE_PATH = "bg_image.jpg"  # optional background image (ignored if missing)
MOVIE_DICT_PATH = "movie_dict.pkl"
SIMILARITY_PATH = "similarity.pkl"
# Prefer an env var for TMDB key; fallback to previous hard-coded key if present.
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "c8f3eaf00fb09a1a9ced6e0a7328eff6")
PLACEHOLDER = "https://via.placeholder.com/300x450?text=No+Image"
# Local fallback poster path (from user's environment)
LOCAL_FALLBACK_POSTER = "/mnt/data/419a744f-467f-4a19-bd76-1f24c28dbdc5.png"

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# --------------------------
# Utilities & caching
# --------------------------
@st.cache_resource
def get_session():
    s = requests.Session()
    s.headers.update({"Accept-Encoding": "gzip, deflate"})
    return s


def set_background(image_path: str):
    """Add base64 background if file exists (silent no-op if not found)."""
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        .movie-container {{
            background-color: rgba(0,0,0,0.72);
            padding: 20px;
            border-radius: 12px;
            margin: 12px 0;
            color: white;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        # ignore if missing or unreadable
        pass


# Try to set background but don't fail if missing
set_background(BG_IMAGE_PATH)


def safe_load_pickle(path: str) -> Optional[object]:
    """Return object from pickle or None on any error."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# Load data (gracefully fallback to a tiny dataframe if needed)
movies_dict = safe_load_pickle(MOVIE_DICT_PATH)
similarity = safe_load_pickle(SIMILARITY_PATH)

if movies_dict is None:
    # minimal fallback dataframe so UI still works
    movies = pd.DataFrame([{"title": "No Movie Available", "movie_id": None} for _ in range(5)])
else:
    movies = pd.DataFrame(movies_dict)
    # ensure required columns
    if "title" not in movies.columns:
        movies["title"] = movies.index.astype(str)
    if "movie_id" not in movies.columns:
        movies["movie_id"] = None


# --------------------------
# TMDB helpers (cached)
# --------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_movie_details(movie_id: Optional[int]) -> dict:
    if not movie_id:
        return {}
    try:
        s = get_session()
        r = s.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=5,
        )
        return r.json() or {}
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster_from_tmdb(movie_id: Optional[int]) -> str:
    if not movie_id:
        return PLACEHOLDER
    try:
        details = fetch_movie_details(movie_id)
        poster_path = details.get("poster_path")
        return f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else PLACEHOLDER
    except Exception:
        return PLACEHOLDER


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_watch_providers(movie_id: Optional[int], country_code: str = "IN") -> dict:
    if not movie_id:
        return {}
    try:
        s = get_session()
        r = s.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers",
            params={"api_key": TMDB_API_KEY},
            timeout=5,
        )
        data = r.json() or {}
        return data.get("results", {}).get(country_code, {})
    except Exception:
        return {}


def format_providers_text(providers: dict) -> str:
    if not providers:
        return "Availability info not found"
    flatrate = providers.get("flatrate", []) or []
    rent = providers.get("rent", []) or []
    buy = providers.get("buy", []) or []
    if flatrate:
        names = ", ".join(p.get("provider_name", "") for p in flatrate)
        return f"Streaming on: {names}"
    if rent or buy:
        return "Available to rent/buy online"
    return "Availability info not found"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_youtube_trailer_url(movie_id: Optional[int], title: str) -> str:
    # try TMDB videos then fallback to youtube search
    if movie_id:
        try:
            s = get_session()
            r = s.get(
                f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
                params={"api_key": TMDB_API_KEY, "language": "en-US"},
                timeout=5,
            )
            data = r.json() or {}
            results = data.get("results", []) or []
            yt = [v for v in results if v.get("site") == "YouTube" and v.get("type") == "Trailer"]
            if yt:
                # prefer "official"
                official = [v for v in yt if "official" in (v.get("name", "")).lower()]
                pick = official[0] if official else yt[0]
                key = pick.get("key")
                if key:
                    return f"https://www.youtube.com/watch?v={key}"
        except Exception:
            pass
    # fallback to YouTube search
    q = requests.utils.quote(f"{title} trailer")
    return f"https://www.youtube.com/results?search_query={q}"


# --------------------------
# Utility: robust recommend()
# --------------------------
def recommend(movie_title: str, top_n: int = 5) -> Tuple[List[str], List[str]]:
    """
    Return up to top_n recommended titles and corresponding poster URLs.
    Robust to many edge-cases:
     - missing similarity matrix
     - non-square / unexpected types
     - duplicates
     - not enough neighbors -> pad with other movies
    """
    # If no similarity matrix or movies, bail out
    if similarity is None:
        return [], []

    # Convert similarity to numpy array if possible
    sim_arr = None
    try:
        sim_arr = np.array(similarity)
    except Exception:
        sim_arr = None

    if sim_arr is None or sim_arr.size == 0:
        return [], []

    # Try find index for movie_title
    try:
        movie_index = int(movies[movies["title"] == movie_title].index[0])
    except Exception:
        return [], []

    # Ensure sim array is the right shape
    if sim_arr.ndim == 1:
        # vector similarity (unlikely): treat as distances to list in same order
        distances = sim_arr
    elif sim_arr.ndim == 2:
        # use row corresponding to movie_index if available
        if movie_index < sim_arr.shape[0]:
            distances = sim_arr[movie_index]
        else:
            return [], []
    else:
        return [], []

    # Ensure distances is 1D numeric
    try:
        distances = np.asarray(distances, dtype=float)
    except Exception:
        return [], []

    # get candidate indices sorted descending by score
    # argsort returns ascending so we reverse
    idx_sorted = np.argsort(distances)[::-1]

    # Remove the original movie index from suggestions (if present)
    idx_sorted = [int(i) for i in idx_sorted if int(i) != movie_index]

    # Deduplicate and keep within bounds
    unique_idxs = []
    seen = set()
    for idx in idx_sorted:
        if idx < 0 or idx >= len(movies):
            continue
        if idx in seen:
            continue
        seen.add(idx)
        unique_idxs.append(idx)
        if len(unique_idxs) >= top_n:
            break

    # If not enough results, pad with random other movies (excluding original)
    if len(unique_idxs) < top_n:
        candidates = [i for i in range(len(movies)) if i != movie_index and i not in set(unique_idxs)]
        np.random.shuffle(candidates)
        for c in candidates:
            unique_idxs.append(int(c))
            if len(unique_idxs) >= top_n:
                break

    rec_titles = []
    rec_posters = []
    for idx in unique_idxs[:top_n]:
        title = str(movies.iloc[idx]["title"])
        rec_titles.append(title)
        mid = movies.iloc[idx].get("movie_id", None)
        poster = fetch_poster_from_tmdb(mid)
        if poster == PLACEHOLDER:
            # prefer local fallback when available
            if os.path.exists(LOCAL_FALLBACK_POSTER):
                poster = LOCAL_FALLBACK_POSTER
        rec_posters.append(poster)
    return rec_titles, rec_posters


# --------------------------
# UI: header / featured cards
# --------------------------
st.markdown(
    """
<div style="padding:12px; background: rgba(0,0,0,0.6); border-radius:10px; color:white;">
  <h1 style="margin:6px 0;">🎬 Movie Recommendation System</h1>
  <div>Pick a movie and get content-based recommendations (5 cards). Streamlit + TMDB + Render friendly.</div>
</div>
""",
    unsafe_allow_html=True,
)

# show up to 3 random featured movies
n_featured = min(3, max(1, len(movies)))
featured = movies.sample(n_featured).reset_index(drop=True)

cols = st.columns(n_featured)
for i, col in enumerate(cols):
    row = featured.loc[i]
    title = row.get("title", f"Movie {i+1}")
    movie_id = row.get("movie_id", None)
    details = fetch_movie_details(movie_id) if movie_id else {}
    rating = details.get("vote_average")
    rating_text = f"⭐ {round(float(rating),1)}/10 (TMDB)" if rating else "⭐ Rating not available"
    genres = details.get("genres", []) or []
    genres_text = ", ".join(g.get("name","") for g in genres) if genres else "Genres not available"
    overview = details.get("overview") or "Overview not available"
    providers = fetch_watch_providers(movie_id, "IN") if movie_id else {}
    providers_text = format_providers_text(providers)
    poster = fetch_poster_from_tmdb(movie_id)
    if poster == PLACEHOLDER and os.path.exists(LOCAL_FALLBACK_POSTER):
        poster = LOCAL_FALLBACK_POSTER
    trailer = fetch_youtube_trailer_url(movie_id, title)
    html = dedent(f"""
    <div style="background: rgba(0,0,0,0.65); padding:12px; border-radius:10px; color:#fff;">
      <h3 style="margin:6px 0;">{title}</h3>
      <a href="{trailer}" target="_blank" rel="noopener noreferrer">
        <img src="{poster}" alt="poster" style="width:100%; max-width:280px; border-radius:8px;">
      </a>
      <div style="font-weight:600; margin-top:8px;">{rating_text}</div>
      <div style="opacity:0.9; margin-top:6px;"><strong>Genres:</strong> {genres_text}</div>
      <div style="opacity:0.9; margin-top:4px;"><strong>Watch:</strong> {providers_text}</div>
      <div style="margin-top:8px; opacity:0.95;">{overview[:180]}{ '...' if len(overview)>180 else ''}</div>
    </div>
    """)
    with col:
        st.markdown(html, unsafe_allow_html=True)


# --------------------------
# Recommendation UI + CSS for 5 cards
# --------------------------
st.markdown(
    """
<style>
.rec-row { width:100%; display:flex; justify-content:space-between; gap:10px; flex-wrap:nowrap; }
.rec-card { background: rgba(0,0,0,0.6); border-radius:12px; padding:12px; width:19%; color:#fff; display:flex; flex-direction:column; align-items:center; box-sizing:border-box; min-height:480px; }
.rec-poster { width:100%; height:300px; object-fit:cover; border-radius:8px; margin-bottom:8px; }
.rec-title { font-weight:700; margin:6px 0; text-align:center; }
.rec-meta { font-size:0.85rem; opacity:0.92; text-align:center; margin-bottom:6px; }
.rec-overview { font-size:0.78rem; opacity:0.9; text-align:left; max-height:100px; overflow:hidden; width:100%; }
@media (max-width:1100px) { .rec-card { width:30%; min-height:420px; } .rec-row { overflow-x:auto; } }
@media (max-width:700px) { .rec-card { width:220px; flex:0 0 auto; } }
</style>
""",
    unsafe_allow_html=True,
)

selected_movie = st.selectbox("Select a movie to get recommendations", movies["title"].tolist())

if st.button("Recommend"):
    names, posters = recommend(selected_movie, top_n=5)
    if not names:
        st.info("No recommendations available for this selection.")
    else:
        # Render 5 cards horizontally (responsive)
        cards = ['<div class="rec-row">']
        for i, name in enumerate(names):
            mid = None
            # safe lookup of movie_id
            try:
                if "movie_id" in movies.columns:
                    mid = movies.loc[movies["title"] == name, "movie_id"].iloc[0]
            except Exception:
                mid = None
            details = fetch_movie_details(mid) if mid else {}
            rating = details.get("vote_average")
            rating_text = f"⭐ {round(float(rating),1)}/10" if rating else "⭐ N/A"
            genres = details.get("genres", []) or []
            genres_text = ", ".join(g.get("name","") for g in genres) if genres else "Genres N/A"
            providers_text = format_providers_text(fetch_watch_providers(mid, "IN") if mid else {})
            overview = details.get("overview") or "Overview not available"
            overview_text = overview[:260] + "..." if len(overview) > 260 else overview
            poster_url = posters[i] if i < len(posters) else LOCAL_FALLBACK_POSTER
            # if poster_url is placeholder but local fallback exists, use it
            if poster_url == PLACEHOLDER and os.path.exists(LOCAL_FALLBACK_POSTER):
                poster_url = LOCAL_FALLBACK_POSTER
            trailer = fetch_youtube_trailer_url(mid, name)
            card_html = dedent(f"""
            <div class="rec-card">
              <a href="{trailer}" target="_blank" rel="noopener noreferrer">
                <img class="rec-poster" src="{poster_url}" alt="poster">
              </a>
              <div class="rec-title">{name}</div>
              <div class="rec-meta">{rating_text} • {genres_text}</div>
              <div class="rec-meta" style="font-size:0.75rem; opacity:0.85;">{providers_text}</div>
              <div class="rec-overview">{overview_text}</div>
            </div>
            """)
            cards.append(card_html.strip())
        cards.append("</div>")
        st.markdown("\n".join(cards), unsafe_allow_html=True)
