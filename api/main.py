import pandas as pd
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

with open('src/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('src/ratings.pkl', 'rb') as f:
    ratings = pickle.load(f)

with open('src/movies.pkl', 'rb') as f:
    movies = pickle.load(f)

app = FastAPI(title="Movie Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Movie Recommendation API is running!"}

@app.get("/genres")
def get_all_genres():
    all_genres = movies['Genres'].str.split('|').explode().unique().tolist()
    all_genres = sorted([g for g in all_genres if g])
    return {"genres": all_genres}

@app.get("/recommend/genre/{genre}")
def recommend_by_genre(genre: str, n: int = 10):
    genre_movies = movies[movies['Genres'].str.contains(genre, case=False, na=False)]

    if genre_movies.empty:
        return {"error": f"No movies found for genre: {genre}"}

    genre_movie_ids = genre_movies['MovieID'].tolist()
    genre_ratings = ratings[ratings['MovieID'].isin(genre_movie_ids)]

    avg_ratings = genre_ratings.groupby('MovieID').agg(
        avg_rating=('Rating', 'mean'),
        num_ratings=('Rating', 'count')
    ).reset_index()

    avg_ratings = avg_ratings[avg_ratings['num_ratings'] >= 50]
    avg_ratings = avg_ratings.sort_values('avg_rating', ascending=False)

    top_n = avg_ratings.head(n)

    results = []
    for _, row in top_n.iterrows():
        movie_row = movies[movies['MovieID'] == row['MovieID']].iloc[0]
        results.append({
            "movie_id": int(row['MovieID']),
            "title": movie_row['Title'],
            "genres": movie_row['Genres'],
            "avg_rating": round(row['avg_rating'], 2),
            "num_ratings": int(row['num_ratings'])
        })

    return {"genre": genre, "movies": results}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10):
    if user_id not in ratings['UserID'].values:
        return {"error": f"User {user_id} not found. Try a user ID between 1 and 6040."}

    rated_movies = ratings[ratings['UserID'] == user_id]['MovieID'].tolist()
    all_movie_ids = movies['MovieID'].tolist()
    unrated = [mid for mid in all_movie_ids if mid not in rated_movies]
    predictions_list = [model.predict(user_id, mid) for mid in unrated]
    predictions_list.sort(key=lambda x: x.est, reverse=True)

    top_n = predictions_list[:n]
    results = []
    for pred in top_n:
        movie_row = movies[movies['MovieID'] == pred.iid].iloc[0]
        results.append({
            "movie_id": int(pred.iid),
            "title": movie_row['Title'],
            "genres": movie_row['Genres'],
            "predicted_rating": round(pred.est, 2)
        })

    return {"user_id": user_id, "recommendations": results}

@app.get("/movie/{movie_id}")
def get_movie(movie_id: int):
    movie = movies[movies['MovieID'] == movie_id]
    if movie.empty:
        return {"error": "Movie not found"}
    row = movie.iloc[0]
    return {
        "movie_id": movie_id,
        "title": row['Title'],
        "genres": row['Genres']
    }