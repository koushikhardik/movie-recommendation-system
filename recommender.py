# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load the datasets
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')

# Merge the dataframes
movies_df = movies_df.merge(credits_df, on='title')

# Select features for the model
features = ['genres', 'keywords', 'overview', 'cast', 'crew']
for feature in features:
    movies_df[feature] = movies_df[feature].fillna('')

# Helper function to clean and extract names from JSON-like data
def get_names(text):
    names = []
    for i in json.loads(text):
        names.append(i['name'].replace(" ", ""))
    return " ".join(names)

def get_director(text):
    for i in json.loads(text):
        if i['job'] == 'Director':
            return i['name'].replace(" ", "")
    return ""

# Apply helper functions to the dataframe
movies_df['genres'] = movies_df['genres'].apply(get_names)
movies_df['keywords'] = movies_df['keywords'].apply(get_names)
movies_df['cast'] = movies_df['cast'].apply(lambda x: " ".join([i['name'].replace(" ", "") for i in json.loads(x)][:3]))
movies_df['crew'] = movies_df['crew'].apply(get_director)

# Create a "soup" of combined text features
movies_df['soup'] = movies_df['overview'] + ' ' + movies_df['keywords'] + ' ' + movies_df['cast'] + ' ' + movies_df['crew'] + ' ' + movies_df['genres']

# Convert text into numerical vectors using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['soup'])

# Calculate similarity using Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies_df[movies_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]

# --- Example Usage ---
# To use this script, you would need the CSV files in the same directory and then run:
# recommendations = get_recommendations('The Dark Knight Rises')
# print(recommendations)
