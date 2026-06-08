import pandas as pd

# Load ratings
ratings = pd.read_csv('data/ratings.dat', sep='::', engine='python',
                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

# Load movies
movies = pd.read_csv('data/movies.dat', sep='::', engine='python',
                     names=['MovieID', 'Title', 'Genres'], encoding='latin-1')

# Load users
users = pd.read_csv('data/users.dat', sep='::', engine='python',
                    names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip'])

# Basic stats
print("=== RATINGS ===")
print(ratings.shape)
print(ratings.head())
print(ratings['Rating'].describe())

print("\n=== MOVIES ===")
print(movies.shape)
print(movies.head())

print("\n=== USERS ===")
print(users.shape)
print(users.head())

print("\n=== Sparsity ===")
sparsity = 1 - (len(ratings) / (ratings['UserID'].nunique() * ratings['MovieID'].nunique()))
print(f"Matrix sparsity: {sparsity:.4%}")