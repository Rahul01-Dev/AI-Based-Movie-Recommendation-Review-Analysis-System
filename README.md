# 🎬 AI-Based Movie Recommendation & Review Analysis System

> **A final-year / internship ML project** built with Python, covering content-based filtering,
> popularity-based ranking, NLP sentiment analysis, and collaborative filtering using real-world datasets.

---

## 📌 Problem Statement

People spend a surprising amount of time just deciding *what* to watch.
Existing streaming platforms use powerful recommendation engines, but most people
don't know how those systems actually work under the hood.

This project builds a simplified but realistic movie recommendation system from scratch using
the **MovieLens dataset** — the same kind of data Netflix and similar platforms use.
It also includes a **review sentiment analyser** to understand whether a movie review is positive or negative,
which is useful for platforms that aggregate user feedback.

The goal is practical: take raw rating + review data, apply standard ML techniques,
and deliver something useful through a web interface.

---

## ✨ Features

| Feature | What It Does |
|---|---|
| 🎯 Content-Based Filtering | Recommends movies with similar genres using TF-IDF + Cosine Similarity |
| 🏆 Popularity-Based Ranking | Ranks movies using IMDB's weighted rating formula (balances ratings + vote count) |
| 💬 Sentiment Analyser | Classifies any movie review as Positive / Negative / Neutral using VADER |
| 📊 Evaluation Explanation | In-app explanation of algorithms, formulas, and model metrics |
| 📁 Research Notebooks | 4 Jupyter notebooks covering EDA, Recommender, Sentiment, and Fake Review Detection |

---

## 📁 Project Structure

```
movie_recommender/
├── data/                    ← Put your CSV files here (see Dataset section)
├── reports/                 ← Auto-created; stores generated chart PNGs
├── notebooks/               ← Research models and Exploratory Data Analysis
│   ├── 1_eda.ipynb              
│   ├── 2_recommender.ipynb      ← SVD + Content-Based + Popularity
│   ├── 3_sentiment.ipynb        ← NLP Sentiment Analysis 
│   └── 4_fake_detection.ipynb   ← Suspicious user detection
├── app.py                   ← Main Streamlit web application
└── requirements.txt         ← Python dependencies
```

---

## 📂 Dataset Setup

### 1. MovieLens Small (100K ratings) — Free

> https://grouplens.org/datasets/movielens/latest/

Download `ml-latest-small.zip`, unzip it, and copy these two files into `data/`:
- `ratings.csv` — (userId, movieId, rating, timestamp)
- `movies.csv`  — (movieId, title, genres)

### 2. IMDb 50K Movie Reviews — Free (Kaggle account needed)

> https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Download and copy into `data/`:
- `IMDB Dataset.csv` — (review, sentiment)

> 💡 The first two notebooks (and the Streamlit app) only need MovieLens data.
> The IMDb dataset is only needed for Notebook 3 (Sentiment) and Notebook 4.

---

## ⚙️ Installation

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Mac / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download VADER lexicon (one-time)
python -c "import nltk; nltk.download('stopwords')"
```

---

## 🚀 How to Run

### Option A — Streamlit Web App (recommended)

```bash
cd movie_recommender
streamlit run app.py
```

Opens at `http://localhost:8501`. No Jupyter needed.

### Option B — Run Notebooks Step by Step

```bash
jupyter notebook
```

Run notebooks in order:

| Notebook | Topic | Approx. Time |
|---|---|---|
| `1_eda.ipynb` | Exploratory Data Analysis | ~1 min |
| `2_recommender.ipynb` | SVD + Content-Based + Popularity | ~3–5 min |
| `3_sentiment.ipynb` | NLP Sentiment Analysis | ~5–10 min |
| `4_fake_detection.ipynb` | Anomaly Detection | ~1 min |

---

## 🧠 How the Algorithms Work

### Content-Based Filtering (TF-IDF + Cosine Similarity)
Each movie's genres are converted into a numeric vector using TF-IDF.
Cosine similarity then finds how "close" two movies are in genre space.
A score of **1.0** = identical genre profile; **0.0** = completely different.

### Popularity-Based Ranking (IMDB Weighted Formula)
Sorting by average rating alone is misleading — a movie with 3 ratings of 5★
shouldn't outrank one with 5,000 ratings of 4.8★.
IMDB's formula: `WS = (v / (v + m)) × R + (m / (v + m)) × C`
balances vote count (`v`) with average rating (`R`) and global mean (`C`).

### Collaborative Filtering (SVD)
The notebook (Notebook 2) uses **Singular Value Decomposition (SVD)** from `scikit-surprise`
to learn hidden patterns in user–movie rating behaviour.
Evaluation uses **RMSE** (~0.87) and **MAE** (~0.67).

### Sentiment Analysis (VADER)
VADER is a rule-based NLP model that assigns a compound score from −1 to +1.
Scores ≥ 0.05 → Positive, ≤ −0.05 → Negative, in between → Neutral.
No training needed — works out-of-the-box on review text.

---

## 📊 Model Performance

| Model | Metric | Typical Value |
|---|---|---|
| SVD (Collaborative Filtering) | RMSE | ~0.87 |
| SVD (Collaborative Filtering) | MAE  | ~0.67 |
| Logistic Regression (Sentiment) | Accuracy | ~89% |
| Naive Bayes (Sentiment) | Accuracy | ~85% |

> *Exact numbers will vary slightly — run the notebooks yourself to reproduce results.*

---

## 🖥️ App Pages

| Page | Description |
|---|---|
| 🎯 Movie Recommender | Pick Content-Based or Popularity-Based; get a ranked table with scores |
| 💬 Sentiment Checker | Paste any review and get Positive / Negative / Neutral with a compound score |
| ℹ️ How It Works | In-app explanation of Cosine Similarity, IMDB formula, VADER, and model metrics |

---

## ☁️ Deployment (Streamlit Cloud)

You can deploy this project for free on Streamlit Cloud in a few steps:

1. Push your project folder to a **public GitHub repo**
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub
3. Click **"New app"** → select your repo → set **Main file path** to `app.py`
4. Add any secrets or environment variables if needed (none required for this project)
5. Hit **Deploy** — your app gets a public URL like `https://your-app.streamlit.app`

> ⚠️ Make sure `requirements.txt` is up to date before deploying.
> The data files (`data/`) should either be included in the repo or fetched at runtime.

---

## 🔮 Future Scope

- **Neural Collaborative Filtering** — replace SVD with a PyTorch-based neural model for better accuracy
- **BERT Sentiment** — use a pre-trained transformer for more nuanced review understanding
- **Movie Poster API** — fetch posters via the TMDB API for visual recommendations
- **Session-Based Recommendations** — track what the user watched/liked in the same session and adapt
- **User Login** — store user rating history to improve personalised recommendations over time
- **A/B Testing Module** — compare different recommendation strategies side by side

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data processing |
| scikit-learn | TF-IDF, Cosine Similarity, Logistic Regression, Isolation Forest |
| scikit-surprise | SVD Collaborative Filtering |
| VADER (vaderSentiment) | Rule-based Sentiment Analysis |
| Streamlit | Web application |
| Matplotlib / Seaborn | Data visualisation in notebooks |

---

## 👨‍💻 About This Project

This project was built as part of an internship/academic ML curriculum.
The aim was not just to follow tutorials, but to actually understand and explain
each algorithm — why it's used, what its limitations are, and how it performs on real data.

The MovieLens dataset is widely used in recommendation system research, which makes
this a good start for understanding how real-world systems like Netflix or YouTube
approach the recommendation problem.
