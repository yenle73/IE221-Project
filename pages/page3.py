import os
import time
import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import streamlit as st

movies_filename = "data/movies.csv"

ratings_filename = "data/ratings.csv"

df_movies = pd.read_csv(
    movies_filename,
    usecols=['movieId', 'title'],
    dtype={'movieId': 'int32', 'title': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

### Pre-processing the data ###
df_movies_cnt = pd.DataFrame(df_ratings.groupby('movieId').size(), columns=['count'])
df_movies_cnt['count'].quantile(np.arange(1, 0.6, -0.05))

# Filter Data to take out only Popular Movies
popularity_thres = 50
popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))
df_ratings_drop_movies = df_ratings[df_ratings.movieId.isin(popular_movies)]

df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('userId').size(), columns=['count'])

# Filter for Inactive Users
ratings_thres = 50
active_users = list(set(df_users_cnt.query('count >= @ratings_thres').index))
df_ratings_drop_users = df_ratings_drop_movies[df_ratings_drop_movies.userId.isin(active_users)]

# Pivot and create movie-user matrix
movie_user_mat = df_ratings_drop_users.pivot(index='movieId', columns='userId', values='rating').fillna(0)
# create mapper from movie title to index
movie_to_idx = {
    movie: i for i, movie in 
    enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title))
}
# transform matrix to scipy sparse matrix
movie_user_mat_sparse = csr_matrix(movie_user_mat.values)

### Building Model ###
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(movie_user_mat_sparse)

def fuzzy_matching(mapper, fav_movie, verbose=True):
    """
    return the closest match via fuzzy ratio. If no match found, return None
    
    Parameters
    ----------    
    mapper: dict, map movie title name to index of the movie in data

    fav_movie: str, name of user input movie
    
    verbose: bool, print log if True

    Return
    ------
    index of the closest match
    """
    match_tuple = []
    # get match
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]
def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):
    model_knn.fit(data)
  #the function below is a helper function defined to check presence of Movie Name
    idx = fuzzy_matching(mapper, fav_movie, verbose=True)

    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
  # get list of raw idx of recommendations
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
  # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
  # print recommendations
    st.markdown(f"<h3 style='text-align: center; color: #10316B;'>Because You Like \"{user_title}\" So You May Also Like</h3>", unsafe_allow_html=True)
    results = {'Rank': 'Movie'}
    for i, (idx, dist) in enumerate(raw_recommends):
    #st.markdown(f"<p >{i+1}. {reverse_mapper[idx]}</p>", unsafe_allow_html=True)
        results[i+1] = reverse_mapper[idx]
    return results


st.markdown("<h2 style='text-align: center; color: #10316B;'>Collaborative Recommender</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #10316B;'>To teach us what you like, please type in a movie that you already know and like.</h5>", unsafe_allow_html=True)

form = st.form(key='my_form')
user_title = form.text_input(label='Enter a movie\'s name')
submit_button = form.form_submit_button(label='Submit')

if submit_button:
    with st.spinner('Searching for movies...'):
        time.sleep(3)
        st.success('Matches Found!')
        st.table(make_recommendation(model_knn=model_knn, data=movie_user_mat_sparse, fav_movie=user_title, mapper=movie_to_idx, n_recommendations=10))