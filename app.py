# AI-Based Movie Recommendation & Review Analysis System
# Streamlit Web App — run with: streamlit run app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 12px 18px;
    }
    thead tr th { background-color: #313244 !important; }
    .sidebar-footer { font-size: 0.78rem; color: #6c7086; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ── Core Logic & Data Handling ───────────────────────────────────────────────

@st.cache_data(show_spinner="Loading and optimizing datasets…")
def load_and_prep_data():
    """Loads datasets and optimizes pandas memory usage."""
    # Load ratings subset to keep memory footprint low
    ratings = pd.read_csv(
        "data/ratings.csv", 
        nrows=100_000, 
        usecols=["userId", "movieId", "rating"], # Drop timestamp implicitly
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"}
    )
    
    # Load movies
    movies = pd.read_csv("data/movies.csv", dtype={"movieId": "int32"})
    
    # Merge for popularity scoring later
    df = pd.merge(ratings, movies, on="movieId", how="left")
    
    # Extract unique global genres for dropdown filters
    all_genres = sorted({
        g for gs in movies["genres"].dropna()
        for g in gs.split("|") if g != "(no genres listed)"
    })
    
    return ratings, movies, df, all_genres


@st.cache_resource(show_spinner="Building content vectors…")
def compute_content_model(movies):
    """Computes and caches the TF-IDF matrix for string similarity."""
    movies = movies.copy()
    movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)
    
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(movies["genres_clean"])
    title_key_map = pd.Series(movies.index, index=movies["title"]).to_dict()
    
    return matrix, title_key_map, movies


@st.cache_data(show_spinner="Calculating popularity scores…")
def compute_popularity_scores(df):
    """Computes IMDB weighted rating for all movies based on ratings distribution."""
    # Group by movie to get vote counts and averages
    stats = df.groupby(["movieId", "title", "genres"])["rating"].agg(
        num_votes="count", avg_rating="mean"
    ).reset_index()
    
    # IMDB Formula components
    global_mean = df["rating"].mean()
    min_votes = stats["num_votes"].quantile(0.50) # 50th percentile threshold
    
    # Calculate weighted score
    stats["weighted_score"] = (
        (stats["num_votes"] / (stats["num_votes"] + min_votes)) * stats["avg_rating"] +
        (min_votes / (stats["num_votes"] + min_votes)) * global_mean
    ).round(4)
    
    return stats


# ── Recommendation Functions ──────────────────────────────────────────────────

def recommend_by_content(title, matrix, title_map, movies_ref, n=10):
    """Returns movies with similar genres using Cosine Similarity."""
    if title not in title_map:
        # Graceful fallback for fuzzy matches
        matches = [t for t in title_map if title.lower() in t.lower()]
        if not matches: return pd.DataFrame(), None
        title = matches[0]
        
    idx = title_map[title]
    similarity_row = cosine_similarity(matrix[idx], matrix)[0]
    
    # Get top N similar excluding the movie itself
    scores = sorted(enumerate(similarity_row), key=lambda x: x[1], reverse=True)
    top_n = [s for s in scores if s[0] != idx][:n]
    
    # Format results
    result = movies_ref.iloc[[i for i, _ in top_n]][["title", "genres"]].copy()
    result["similarity_score"] = [round(s, 4) for _, s in top_n]
    result.columns = ["Title", "Genres", "Similarity Score"]
    
    return result.reset_index(drop=True), title


def recommend_by_popularity(stats_df, genre="All Genres", n=10):
    """Returns top movies using the pre-computed weighted score."""
    result = stats_df.copy()
    if genre != "All Genres":
        result = result[result["genres"].str.contains(genre, case=False, na=False)]
        
    result = result.sort_values("weighted_score", ascending=False).head(n)
    
    # Format results
    out = result[["title", "genres", "num_votes", "avg_rating", "weighted_score"]]
    out.columns = ["Title", "Genres", "Votes", "Avg Rating", "Weighted Score"]
    
    return out.reset_index(drop=True)


# ── NLP Functions ────────────────────────────────────────────────────────────

@st.cache_resource
def get_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

def analyze_review_sentiment(text, analyzer):
    """Classifies text sentiment using VADER."""
    compound = analyzer.polarity_scores(text)["compound"]
    if compound >= 0.05:    return "Positive 😊", compound
    elif compound <= -0.05: return "Negative 😞", compound
    else:                   return "Neutral 😐",  compound


# ── UI Bootstrap ──────────────────────────────────────────────────────────────

ratings, movies, combined_df, all_genres = load_and_prep_data()
tfidf_matrix, title_to_idx, movies_local = compute_content_model(movies)
movie_stats = compute_popularity_scores(combined_df)
vader_model = get_sentiment_analyzer()
all_titles = sorted(movies["title"].dropna().tolist())


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 AI Movie Recommender")
    st.caption("AI-Based Movie Recommendation & Review Analysis System")
    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        ["🎯 Movie Recommender", "💬 Sentiment Checker", "ℹ️ How It Works"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**📊 Global Dataset Stats**")
    col1, col2 = st.columns(2)
    col1.metric("Movies", f"{movies['movieId'].nunique():,}")
    col2.metric("Users",  f"{ratings['userId'].nunique():,}")
    st.metric("Total Ratings Processed", f"{len(ratings):,}")

    st.markdown("---")
    st.markdown(
        '<p class="sidebar-footer">Built with ❤️ for Academic Submission<br>'
        'Powered by Python, Streamlit & Scikit-Learn</p>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: MOVIE RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
if selected_page == "🎯 Movie Recommender":
    st.title("🎯 Movie Recommendation Engine")
    st.markdown("Generate targeted movie suggestions using Content-Based or Popularity-Based ML filtering.")
    st.markdown("---")

    rec_type = st.selectbox(
        "Algorithm Selection:",
        ["Content-Based — Find by genre similarity (Cosine Similarity)", 
         "Popularity-Based — Find top-rated (IMDB Weighted Rating)"]
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Content-Based View ──
    if rec_type.startswith("Content"):
        st.subheader("🔍 Suggest similar movies to...")
        col_a, col_b = st.columns([3, 1])
        
        with col_a:
            selected_movie = st.selectbox("Movie selection", [""] + all_titles, label_visibility="collapsed")
        with col_b:
            n_recs = st.number_input("Count", min_value=5, max_value=20, value=10, step=5, label_visibility="collapsed")

        if st.button("🎬 Run Recommender", use_container_width=True):
            if not selected_movie:
                st.warning("⚠️ Please select a baseline movie first.")
            else:
                with st.spinner("Computing cosine similarity vectors…"):
                    results, matched_name = recommend_by_content(
                        selected_movie, tfidf_matrix, title_to_idx, movies_local, n=n_recs
                    )
                
                if matched_name is None:
                    st.error(f'❌ Failed to locate "{selected_movie}" in the vector space.')
                else:
                    st.success(f"✅ Displaying top {len(results)} movies structurally similar to **{matched_name}**")
                    st.dataframe(
                        results.style.background_gradient(subset=["Similarity Score"], cmap="Blues"),
                        use_container_width=True,
                    )

    # ── Popularity-Based View ──
    else:
        st.subheader("🏆 Global Top Rated")
        col_a, col_b = st.columns([3, 1])
        
        with col_a:
            genre_filter = st.selectbox("Genre constraint", ["All Genres"] + all_genres, label_visibility="collapsed")
        with col_b:
            n_recs = st.number_input("Count", min_value=5, max_value=20, value=10, step=5, label_visibility="collapsed")

        if st.button("🏅 Fetch Top Movies", use_container_width=True):
            with st.spinner("Filtering weighted rating matrices…"):
                results = recommend_by_popularity(movie_stats, genre=genre_filter, n=n_recs)
                
            if results.empty:
                st.warning("No data points found matching that constraint.")
            else:
                st.success(f"✅ Top {len(results)} evaluated movies against constraint: **{genre_filter}**")
                st.dataframe(
                    results.style.background_gradient(subset=["Weighted Score"], cmap="Greens"),
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: SENTIMENT CHECKER
# ══════════════════════════════════════════════════════════════════════════════
elif selected_page == "💬 Sentiment Checker":
    st.title("💬 Review Sentiment Analyser")
    st.markdown("Classifies unstructured review texts as Positive, Negative, or Neutral using the **VADER** NLP algorithm.")
    st.markdown("---")

    user_review = st.text_area(
        "Input unstructured text:",
        height=180,
        placeholder="Paste a movie review here to compute its underlying sentiment vector..."
    )

    col1, col2 = st.columns([1, 3])
    if col1.button("🔍 Run NLP Model", use_container_width=True):
        if not user_review.strip():
            st.warning("⚠️ Text corpus is empty. Please provide input.")
        else:
            sentiment_label, compound_val = analyze_review_sentiment(user_review, vader_model)
            st.markdown("---")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Sentiment", sentiment_label)
            c2.metric("Compound Polarity", f"{compound_val:.4f}", help="Range: [-1.0, 1.0]")
            c3.metric("Token Count", f"{len(user_review.split())} words")

            if "Positive" in sentiment_label:
                st.success("Analysis complete. The model detected an overall **positive** emotional valence.")
            elif "Negative" in sentiment_label:
                st.error("Analysis complete. The model detected an overall **negative** emotional valence.")
            else:
                st.info("Analysis complete. The model detected neutralized or conflicting valence.")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: EXPLANATION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
elif selected_page == "ℹ️ How It Works":
    st.title("ℹ️ Technical Model Documentation")
    st.markdown("Core mathematical concepts utilized in this deployment.")
    st.markdown("---")

    with st.expander("📐 Cosine Similarity — Content-Based", expanded=True):
        st.markdown("""
**Objective:** Measure structural similarity between two movie genre vectors.

**Implementation:** Movie genres are tokenised and mapped into high-dimensional space using **TF-IDF** (Term Frequency-Inverse Document Frequency). Cosine similarity measures the angle between these vectors irrespective of magnitude.

**Formula:** `similarity(A, B) = (A · B) / (||A|| × ||B||)`
        """)

    with st.expander("🏅 IMDB Weighted Rating — Popularity"):
        st.markdown("""
**Objective:** Adjust pure mean ratings to factor in statistical confidence (number of votes).

**Formula:** `WS = (v / (v + m)) × R + (m / (v + m)) × C`
- `v`: Valid votes
- `m`: Threshold limit (50th percentile)
- `R`: Local mean rating
- `C`: Global mean rating

This punishes movies with an unrealistically high average based on low participant counts.
        """)

    with st.expander("💬 VADER NLP Classifier"):
        st.markdown("""
**Objective:** Execute semantic rule-based sentiment reasoning without relying on large hardware-constrained transformers.

**Implementation:** Relies on a pre-computed lexicon dataset mapped to sentiment poles, amplified dynamically by text heuristics like punctuation intensity or capitalisation.
        """)
