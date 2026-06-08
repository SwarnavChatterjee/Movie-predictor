import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load ratings
ratings = pd.read_csv('data/ratings.dat', sep='::', engine='python',
                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

# Load movies
movies = pd.read_csv('data/movies.dat', sep='::', engine='python',
                     names=['MovieID', 'Title', 'Genres'], encoding='latin-1')

# Prepare data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['UserID', 'MovieID', 'Rating']], reader)

# Train/test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train SVD model
print("Training SVD model...")
model = SVD(n_factors=100, n_epochs=20, random_state=42)
model.fit(trainset)

# Evaluate
predictions = model.test(testset)
print(f"RMSE: {accuracy.rmse(predictions, verbose=False):.4f}")
print(f"MAE:  {accuracy.mae(predictions, verbose=False):.4f}")

# Recommendation function
def get_top_n_recommendations(user_id, n=10):
    rated_movies = ratings[ratings['UserID'] == user_id]['MovieID'].tolist()
    all_movie_ids = movies['MovieID'].tolist()
    unrated = [mid for mid in all_movie_ids if mid not in rated_movies]

    predictions_list = [model.predict(user_id, mid) for mid in unrated]
    predictions_list.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions_list[:n]

    print(f"\nTop {n} recommendations for User {user_id}:")
    print("-" * 50)
    for i, pred in enumerate(top_n, 1):
        title = movies[movies['MovieID'] == pred.iid]['Title'].values[0]
        genres = movies[movies['MovieID'] == pred.iid]['Genres'].values[0]
        print(f"{i}. {title} ({genres}) — Predicted: {pred.est:.2f}")

get_top_n_recommendations(user_id=1, n=10)

import pickle


with open('src/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('src/ratings.pkl', 'wb') as f:
    pickle.dump(ratings, f)

with open('src/movies.pkl', 'wb') as f:
    pickle.dump(movies, f)

print("\nModel saved successfully!")