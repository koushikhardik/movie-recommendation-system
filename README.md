# AI-Powered Movie Recommendation System

## Overview

This project is a content-based movie recommendation system built to suggest movies to users based on their similarity. Given a movie title, the system provides a list of the top 10 most similar movies. This was developed as a portfolio project to demonstrate skills in machine learning, data processing, and feature extraction.

## Technologies Used

* **Language:** Python
* **Libraries:**
    * Pandas: For data manipulation and analysis.
    * Scikit-learn: For implementing TF-IDF Vectorization and Cosine Similarity.

## How It Works

1.  **Data Loading & Preprocessing:** The system loads the TMDB 5000 movie dataset, merges the relevant movie and credit information, and cleans the text data (genres, keywords, cast, etc.).
2.  **Feature Engineering:** A "soup" of important features (overview, keywords, cast, director, genres) is created for each movie.
3.  **Vectorization:** The text "soup" is converted into a matrix of numerical vectors using `TfidfVectorizer`, which helps in quantifying the importance of different words.
4.  **Similarity Calculation:** `Cosine Similarity` is used to calculate a similarity score between every movie and every other movie based on their vector representations.
5.  **Recommendation:** When a user inputs a movie title, the system retrieves the top 10 movies with the highest similarity scores.

## Dataset

This project uses the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) available on Kaggle.

## How to Run

1.  Ensure Python and the required libraries (pandas, scikit-learn) are installed.
2.  Place the `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` files in the same directory as the script.
3.  Run the `recommender.py` script. The example usage at the bottom of the file can be modified to get recommendations for different movies.
4.  
