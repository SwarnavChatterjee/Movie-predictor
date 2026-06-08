# 🎬 Movie Recommendation System

An end-to-end movie recommendation system built on the MovieLens 1M dataset using SVD-based collaborative filtering. The system predicts personalized movie recommendations for users and also supports genre-based browsing.

---

## 📊 Dataset

- **Source:** [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
- **Size:** 1 million ratings from 6,040 users on 3,883 movies
- **Matrix Sparsity:** 95.53%

---

## 🧠 Model

- **Algorithm:** SVD (Singular Value Decomposition) — Matrix Factorization
- **Library:** Scikit-Surprise
- **Latent Factors:** 100
- **Epochs:** 20
- **Train/Test Split:** 80/20

### Results
| Metric | Score |
|--------|-------|
| RMSE | 0.8729 |
| MAE | 0.6845 |

---

## 🏗️ Project Structure
ML project/
├── data/
│   ├── ratings.dat
│   ├── movies.dat
│   └── users.dat
├── src/
│   ├── eda.py
│   └── model.py
├── api/
│   └── main.py
├── frontend/
│   └── index.html
└── requirements.txt

---

## 🚀 How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/model.py
```

### 3. Start the API
```bash
uvicorn api.main:app --reload
```

### 4. Open the frontend
Open `frontend/index.html` in your browser.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/recommend/{user_id}` | Top 10 recommendations for a user |
| GET | `/recommend/genre/{genre}` | Top 10 movies by genre |
| GET | `/genres` | List all available genres |
| GET | `/movie/{movie_id}` | Get movie details |

---

## 🖥️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | Scikit-Surprise (SVD) |
| Data | Pandas, NumPy |
| API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |

---

## 💡 Key Concepts

- **Collaborative Filtering** — Recommends based on user behavior, not content metadata
- **Matrix Factorization** — Decompresses sparse rating matrix into dense user and movie embeddings
- **Cold Start Problem** — New users fall back to genre-based popularity recommendations
- **Sparsity** — 95.5% of the user-movie matrix is empty; SVD handles this gracefully

---

## 👤 Author

**Swarnav Chatterjee**  

