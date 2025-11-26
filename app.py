# app.py
# Streamlit Movie Recommendation App with external (Google Drive) pickle download
# Save this as app.py and deploy to Streamlit Cloud. Keep only code + small assets in GitHub.

import streamlit as st
import pickle
import pandas as pd
import requests
import base64
import os
import math
from typing import Optional
from textwrap import dedent

# --------------------------
# Config / Constants
# --------------------------
BG_IMAGE_PATH = "bg_image.jpg"  # optional background image
MOVIE_DICT_PATH = "movie_dict.pkl"
SIMILARITY_PATH = "similarity.pkl"

# Google Drive direct-download URLs (using your file IDs)
MOVIE_DICT_URL = "https://drive.google.com/uc?export=download&id=1S7Y4ST7X8BRQVPRDVixyU3n_mbZ3nOsZ"
SIMILARITY_URL = "https://drive.google.com/uc?export=download&id=1keoUuW2bcNzndyfet_gUsKdB3tlLFrz7"

TMDB_API_KEY = "c8f3eaf00fb09a1a9ced6e0a7328eff6"  # for production move to st.secrets
PLACEHOLDER = "https://via.placeholder.com/300x450?text=No+Image"

# Local fallback poster (upload this file in Streamlit Cloud "Files" if using there)
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
    """Set app background from a local image file (base64-embedded)."""
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
        .movie-card-title {{
            margin: 0 0 6px 0;
        }}
        .movie-rating {{
            font-weight: 600;
        }}
        .movie-meta {{
            font-size: 0.85rem;
            opacity: 0.9;
            margin-top: 4px;
        }}
        .movie-overview {{
            font-size: 0.8rem;
            margin-top: 6px;
        }}
        </style>
        """
        st.markdown(background_css, unsafe_allow_html=True)
    except FileNotFoundError:
        # If background image missing, ignore
        pass

set_background(BG_IMAGE_PATH)

# --------------------------
# Download-on-first-run helper
# --------------------------
def download_if_missing(url: str, dest_path: str):
    """Download file from URL if dest_path does not exist."""
    if os.path.exists(dest_path):
        return
    try:
        with st.spinner(f"Downloading {os.path.basename(dest_path)}..."):
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        st.error(f"Failed to download {dest_path}: {e}")

# --------------------------
# Safe pickle loader
# --------------------------
def safe_load_pickle(path: str) -> Optional[object]:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

# Ensure pickles exist locally (download from Drive if needed)
download_if_missing(MOVIE_DICT_URL, MOVIE_DICT_PATH)
download_if_missing(SIMILARITY_URL, SIMILARITY_PATH)

movies_dict = safe_load_pickle(MOVIE_DICT_PATH)
similarity = safe_load_pickle(SIMILARITY_PATH)

if movies_dict is None:
    movies = pd.DataFrame([{"title": "No Movie Available", "movie_id": None}] * 3)
else:
    movies = pd.DataFrame(movies_dict)
    if "title" not in movies.columns:
        movies["title"] = movies.index.astype(str)
    if "movie_id" not in movies.columns:
        movies["movie_id"] = None

# --------------------------
# TMDB fetch helpers (cached)
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
            timeout=4,
        )
        return resp.json() or {}
    except Exception:
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
            timeout=4,
        )
        data = resp.json() or {}
        results = data.get("results", {})
        return results.get(country_code, {})
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

def fetch_youtube_trailer_url(movie_id: Optional[int], title: str) -> str:
    if not movie_id:
        q = requests.utils.quote(f"{title} trailer")
        return f"https://www.youtube.com/results?search_query={q}"
    try:
        s = get_session()
        resp = s.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=4,
        )
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
# Top header and three featured movies
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

if movies is None or movies.empty:
    top_movies = pd.DataFrame([{"title": "No Movie Available", "movie_id": None}] * 3)
else:
    n = min(3, len(movies))
    try:
        top_movies = movies.sample(n).reset_index(drop=True)
    except Exception:
        top_movies = movies.head(n).reset_index(drop=True)
    if len(top_movies) < 3:
        missing = 3 - len(top_movies)
        padding = pd.DataFrame([{"title": "No Movie Available", "movie_id": None}] * missing)
        top_movies = pd.concat([top_movies, padding], ignore_index=True)

col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]

for i, col in enumerate(cols):
    row = top_movies.loc[i]
    title = row.get("title", f"Movie {i+1}")
    movie_id = row.get("movie_id", None)

    details = fetch_movie_details(movie_id) if movie_id else {}
    rating = details.get("vote_average")
    rating_text = f"⭐ {round(float(rating), 1)}/10 (TMDB)" if rating else "⭐ Rating not available"

    genres_list = details.get("genres", []) if details else []
    genres_text = ", ".join(g.get("name", "") for g in genres_list) if genres_list else "Genres not available"
    overview = details.get("overview") if details else None
    overview_text = overview[:180] + "..." if overview and len(overview) > 180 else (overview or "Overview not available")

    providers = fetch_watch_providers(movie_id, "IN") if movie_id else {}
    providers_text = format_providers_text(providers)

    poster_url = fetch_poster_from_tmdb(movie_id)
    if not poster_url or poster_url == PLACEHOLDER:
        poster_url = LOCAL_FALLBACK_POSTER

    trailer_url = fetch_youtube_trailer_url(movie_id, title)

    card_html = f"""
<div class="movie-container">
    <h3 class="movie-card-title">{title}</h3>
    <a href="{trailer_url}" target="_blank" rel="noopener noreferrer">
        <img src="{poster_url}" alt="poster"
             style="width:100%; max-width:280px; border-radius:8px; display:block; margin-bottom:8px;">
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
# Recommendation UI (5 cards full width)
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
.rec-title {
    font-weight: 700;
    margin: 6px 0 4px 0;
    text-align: center;
    font-size: 1rem;
}
.rec-meta {
    font-size: 0.85rem;
    opacity: 0.92;
    text-align: center;
    margin-bottom: 6px;
}
.rec-overview {
    font-size: 0.78rem;
    opacity: 0.9;
    text-align: left;
    max-height: 100px;
    overflow: hidden;
    width: 100%;
}
.rec-link {
    text-decoration: none;
    color: inherit;
}
@media (max-width: 1200px) {
    .rec-card { width: 30%; min-height: 480px; }
}
@media (max-width: 900px) {
    .rec-row {
        overflow-x: auto;
        padding-bottom: 8px;
    }
    .rec-card {
        min-width: 200px;
        width: 200px;
        flex: 0 0 auto;
        min-height: 420px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------
# Robust recommend() (handles weird similarity shapes)
# --------------------------
def recommend(movie: str):
    """Return list of up to 5 recommended titles and poster URLs."""
    if similarity is None or movies is None or movies.empty:
        return [], []

    try:
        idxs = movies.index[movies["title"] == movie].tolist()
        if not idxs:
            return [], []
        movie_index = idxs[0]
    except Exception:
        return [], []

    try:
        row = similarity[movie_index]
        if (hasattr(row, "_len_") and len(row) == len(movies)) or isinstance(row, (list, tuple)):
            distances = list(enumerate(row))
        else:
            try:
                inner = row[0]
                distances = list(enumerate(inner))
            except Exception:
                distances = list(enumerate(list(row)))
    except Exception:
        return [], []

    cleaned = []
    for i, s in distances:
        try:
            if s is None:
                continue
            val = float(s)
            if math.isnan(val):
                continue
            cleaned.append((i, val))
        except Exception:
            continue

    cleaned = sorted(cleaned, key=lambda x: x[1], reverse=True)
    cleaned = [t for t in cleaned if t[0] != movie_index]
    top_n = cleaned[:5]

    rec_titles = []
    rec_posters = []
    for idx, score in top_n:
        if idx < 0 or idx >= len(movies):
            continue
        title = movies.iloc[idx].get("title", f"Movie {idx}")
        rec_titles.append(title)

        movie_id = movies.iloc[idx].get("movie_id", None) if "movie_id" in movies.columns else None
        poster = fetch_poster_from_tmdb(movie_id)
        if not poster or poster == PLACEHOLDER:
            poster = LOCAL_FALLBACK_POSTER
        rec_posters.append(poster)

    return rec_titles, rec_posters

# --------------------------
# Recommendation controls
# --------------------------
unique_titles = movies["title"].fillna("").astype(str).tolist()
selected_movie_name = st.selectbox("Select a movie to get recommendations", unique_titles)

if st.button("Recommend"):
    names, posters = recommend(selected_movie_name)
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
            rating_text = f"⭐ {round(float(rating), 1)}/10" if rating else "⭐ N/A"

            genres_list = details.get("genres", []) if details else []
            genres_text = ", ".join(g.get("name", "") for g in genres_list) if genres_list else "Genres N/A"

            providers = fetch_watch_providers(rec_movie_id, "IN") if rec_movie_id else {}
            providers_text = format_providers_text(providers)

            overview = details.get("overview") if details else None
            overview_text = overview[:260] + "..." if overview and len(overview) > 260 else (overview or "Overview not available")

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
