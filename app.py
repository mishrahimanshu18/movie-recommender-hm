# movie_recommendation_app.py
# Full final single-file Streamlit app with fixes to recommendation indexing + fallbacks

import streamlit as st
import pickle
import pandas as pd
import requests
import base64
import numpy as np
from typing import Optional
from textwrap import dedent

# --------------------------
# Config / Constants
# --------------------------
BG_IMAGE_PATH = "bg_image.jpg"  # optional background image
MOVIE_DICT_PATH = "movie_dict.pkl"
SIMILARITY_PATH = "similarity.pkl"
TMDB_API_KEY = "c8f3eaf00fb09a1a9ced6e0a7328eff6"  # consider moving to env var
PLACEHOLDER = "https://via.placeholder.com/300x450?text=No+Image"
LOCAL_FALLBACK_POSTER = "/mnt/data/419a744f-467f-4a19-bd76-1f24c28dbdc5.png"

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# --------------------------
# Session helper
# --------------------------
@st.cache_resource
def get_session():
    s = requests.Session()
    s.headers.update({"Accept-Encoding": "gzip, deflate"})
    return s

# --------------------------
# Background setter
# --------------------------
def set_background(image_path: str):
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
        b64_encoded = base64.b64encode(img_data).decode()
        background_css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        .movie-container {{
            background-color: rgba(0, 0, 0, 0.72);
            padding: 20px;
            border-radius: 12px;
            margin: 12px 0;
            color: white;
        }}
        .movie-card-title {{ margin: 0 0 6px 0; }}
        .movie-rating {{ font-weight: 600; }}
        .movie-meta {{ font-size: 0.85rem; opacity: 0.9; margin-top: 4px; }}
        .movie-overview {{ font-size: 0.8rem; margin-top: 6px; }}
        </style>
        """
        st.markdown(background_css, unsafe_allow_html=True)
    except FileNotFoundError:
        pass

set_background(BG_IMAGE_PATH)

# --------------------------
# Safe pickle loader
# --------------------------
def safe_load_pickle(path: str) -> Optional[object]:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

movies_dict = safe_load_pickle(MOVIE_DICT_PATH)
similarity = safe_load_pickle(SIMILARITY_PATH)

# Build movies DataFrame safely
if movies_dict is None:
    movies = pd.DataFrame([{"title": "No Movie Available", "movie_id": None}] * 3)
else:
    # movies_dict could be a list of dicts or dict-of-lists; convert carefully
    try:
        movies = pd.DataFrame(movies_dict)
    except Exception:
        # fallback: if it's not directly convertible
        movies = pd.DataFrame(list(movies_dict)) if isinstance(movies_dict, (list, tuple)) else pd.DataFrame([movies_dict])

    # Ensure columns exist
    if "title" not in movies.columns:
        movies["title"] = movies.index.astype(str)
    if "movie_id" not in movies.columns:
        movies["movie_id"] = None

# Reset index to ensure 0..N-1 indexing (important when using similarity matrix indices)
movies = movies.reset_index(drop=True)

# --------------------------
# TMDB helpers (cached)
# --------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_movie_details(movie_id: Optional[int]) -> dict:
    if not movie_id:
        return {}
    try:
        s = get_session()
        resp = s.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json() or {}
    except Exception:
        pass
    return {}

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_poster_from_tmdb(movie_id: Optional[int]) -> str:
    if not movie_id:
        return PLACEHOLDER
    try:
        details = fetch_movie_details(movie_id)
        poster_path = details.get("poster_path")
        return "https://image.tmdb.org/t/p/w500" + poster_path if poster_path else PLACEHOLDER
    except Exception:
        return PLACEHOLDER

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_watch_providers(movie_id: Optional[int], country_code: str = "IN") -> dict:
    if not movie_id:
        return {}
    try:
        s = get_session()
        resp = s.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers",
            params={"api_key": TMDB_API_KEY},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json() or {}
            results = data.get("results", {})
            return results.get(country_code, {})
    except Exception:
        pass
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

def fetch_youtube_trailer_url(movie_id: Optional[int], title: str) -> str:
    if not movie_id:
        q = requests.utils.quote(f"{title} trailer")
        return f"https://www.youtube.com/results?search_query={q}"
    try:
        s = get_session()
        resp = s.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json() or {}
            results = data.get("results", []) or []
            yt_trailers = [v for v in results if v.get("site") == "YouTube" and v.get("type") == "Trailer"]
            official = [v for v in yt_trailers if "official" in (v.get("name", "")).lower()]
            pick = (official[0] if official else (yt_trailers[0] if yt_trailers else (results[0] if results else None)))
            if pick and pick.get("site") == "YouTube" and pick.get("key"):
                return f"https://www.youtube.com/watch?v={pick['key']}"
    except Exception:
        pass
    q = requests.utils.quote(f"{title} trailer")
    return f"https://www.youtube.com/results?search_query={q}"

# --------------------------
# Header + featured movies
# --------------------------
st.markdown(
    """
<div class="movie-container">
    <h1>🎬 Movie Recommendation System</h1>
    <p>Discover your next favorite movie</p>
</div>
""",
    unsafe_allow_html=True,
)

# Top 3 featured (safe)
if movies is None or movies.empty:
    top_movies = pd.DataFrame([{"title": "No Movie Available", "movie_id": None}] * 3)
else:
    n = min(3, len(movies))
    # sample deterministic so UI doesn't jump too wildly
    top_movies = movies.sample(n=n, replace=False, random_state=42).reset_index(drop=True)
    if len(top_movies) < 3:
        missing = 3 - len(top_movies)
        padding = pd.DataFrame([{"title": "No Movie Available", "movie_id": None}] * missing)
        top_movies = pd.concat([top_movies, padding], ignore_index=True)

cols = st.columns(3)
for i, col in enumerate(cols):
    row = top_movies.loc[i]
    title = row.get("title", f"Movie {i+1}")
    movie_id = row.get("movie_id", None)
    details = fetch_movie_details(movie_id) if movie_id else {}
    rating = details.get("vote_average")
    rating_text = f"⭐ {round(float(rating),1)}/10 (TMDB)" if rating else "⭐ Rating not available"
    genres = details.get("genres", []) if details else []
    genres_text = ", ".join(g.get("name", "") for g in genres) if genres else "Genres not available"
    overview = details.get("overview") if details else None
    overview_text = (overview[:180] + "...") if overview and len(overview)>180 else (overview or "Overview not available")
    providers_text = format_providers_text(fetch_watch_providers(movie_id, "IN")) if movie_id else "Availability info not found"
    poster = fetch_poster_from_tmdb(movie_id)
    if poster == PLACEHOLDER:
        poster = LOCAL_FALLBACK_POSTER
    trailer_url = fetch_youtube_trailer_url(movie_id, title)
    card_html = f"""
<div class="movie-container">
  <h3 class="movie-card-title">{title}</h3>
  <a href="{trailer_url}" target="_blank" rel="noopener noreferrer">
    <img src="{poster}" alt="poster" style="width:100%; max-width:280px; border-radius:8px; display:block; margin-bottom:8px;">
  </a>
  <div class="movie-rating">{rating_text}</div>
  <div class="movie-meta"><strong>Genres:</strong> {genres_text}</div>
  <div class="movie-meta"><strong>Watch:</strong> {providers_text}</div>
  <div class="movie-overview">{overview_text}</div>
</div>
"""
    with col:
        st.markdown(card_html, unsafe_allow_html=True)

# --------------------------
# Styles for recommendation cards
# --------------------------
st.markdown(
    """
<style>
.rec-row {
    width: 100%;
    display: flex;
    justify-content: space-between;
    gap: 0px;
    flex-wrap: nowrap;
    padding: 15px 0;
    box-sizing: border-box;
}
.rec-card {
    background: rgba(0,0,0,0.65);
    border-radius: 12px;
    padding: 12px;
    width: 19%;
    box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    color: #fff;
    display: flex;
    flex-direction: column;
    align-items: center;
    box-sizing: border-box;
    min-height: 520px;
}
.rec-poster {
    width: 100%;
    height: 300px;
    object-fit: cover;
    border-radius: 8px;
    margin-bottom: 10px;
}
.rec-title { font-weight:700; margin:6px 0 4px 0; text-align:center; font-size:1rem; }
.rec-meta { font-size:0.85rem; opacity:0.92; text-align:center; margin-bottom:6px; }
.rec-overview { font-size:0.78rem; opacity:0.9; text-align:left; max-height:100px; overflow:hidden; width:100%; }
.rec-link { text-decoration:none; color:inherit; }
@media (max-width:1200px) { .rec-card { width: 30%; min-height:480px; } }
@media (max-width:900px) {
  .rec-row { overflow-x:auto; padding-bottom:8px; }
  .rec-card { min-width:200px; width:200px; flex: 0 0 auto; min-height:420px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------
# Recommendation logic (fixed)
# --------------------------
def recommend(movie_title: str, top_k: int = 5):
    """
    Return (titles_list, posters_list) for top_k recommendations.
    Uses numpy argsort to pick highest similarity indices safely.
    """
    if similarity is None or movies is None or movies.empty:
        return [], []
    # ensure similarity is numpy array
    try:
        sim = np.array(similarity)
    except Exception:
        # try to convert list-of-lists etc
        sim = np.array(similarity) if similarity is not None else None
    if sim is None or sim.ndim != 2:
        return [], []

    # find movie index
    try:
        movie_index = movies[movies["title"] == movie_title].index[0]
    except Exception:
        return [], []

    # sanity: if sim shape doesn't match movies length, abort
    if sim.shape[0] != len(movies):
        return [], []

    distances = sim[movie_index]
    # argsort descending: indices of most similar first
    sorted_idx = np.argsort(-distances)

    # exclude the movie itself and pick top_k unique indices
    rec_indices = [int(i) for i in sorted_idx if int(i) != int(movie_index)]
    rec_indices = rec_indices[:top_k]

    rec_titles = []
    rec_posters = []
    for idx in rec_indices:
        if idx < 0 or idx >= len(movies):
            continue
        title = movies.iloc[idx]["title"]
        rec_titles.append(title)
        movie_id = movies.iloc[idx].get("movie_id", None)
        poster = fetch_poster_from_tmdb(movie_id)
        if poster == PLACEHOLDER:
            poster = LOCAL_FALLBACK_POSTER
        rec_posters.append(poster)
    return rec_titles, rec_posters

# --------------------------
# Controls & display
# --------------------------
# selectbox values: ensure unique titles and stable order
titles = movies["title"].astype(str).tolist()
selected_movie_name = st.selectbox("Select a movie to get recommendations", titles, index=0)

if st.button("Recommend"):
    names, posters = recommend(selected_movie_name, top_k=5)
    if not names:
        st.info("No recommendations available.")
    else:
        card_pieces = ['<div class="rec-row">']
        for idx, name in enumerate(names):
            rec_movie_id = None
            if "movie_id" in movies.columns and not movies[movies["title"] == name].empty:
                rec_movie_id = movies[movies["title"] == name]["movie_id"].iloc[0]

            details = fetch_movie_details(rec_movie_id) if rec_movie_id else {}
            rating = details.get("vote_average")
            rating_text = f"⭐ {round(float(rating),1)}/10" if rating else "⭐ N/A"
            genres_list = details.get("genres", []) if details else []
            genres_text = ", ".join(g.get("name", "") for g in genres_list) if genres_list else "Genres N/A"
            providers_text = format_providers_text(fetch_watch_providers(rec_movie_id, "IN")) if rec_movie_id else "Availability info not found"
            overview = details.get("overview") if details else None
            overview_text = (overview[:260] + "...") if overview and len(overview) > 260 else (overview or "Overview not available")
            poster_url = posters[idx] if idx < len(posters) else LOCAL_FALLBACK_POSTER
            trailer_url = fetch_youtube_trailer_url(rec_movie_id, name)

            card_html = dedent(f"""\ 
            <div class="rec-card">
                <a class="rec-link" href="{trailer_url}" target="_blank" rel="noopener noreferrer">
                    <img class="rec-poster" src="{poster_url}" alt="poster">
                </a>
                <div class="rec-title">{name}</div>
                <div class="rec-meta">{rating_text} • {genres_text}</div>
                <div class="rec-meta" style="font-size:0.75rem; opacity:0.85;">{providers_text}</div>
                <div class="rec-overview">{overview_text}</div>
            </div>
            """)
            card_pieces.append(card_html.strip())
        card_pieces.append("</div>")
        cards_html = "\n".join(card_pieces)
        st.markdown(cards_html, unsafe_allow_html=True)

