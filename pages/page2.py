import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfTransformer #To conduct Term Frequency Inverse Document Frequency Tranformation
from sklearn.feature_extraction.text import CountVectorizer #To create text document matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import plotly.tools as pytools
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import streamlit as st

nltk.download('stopwords')
nltk.download('punkt')

df_movies = pd.read_csv('final_data.csv')
movies = df_movies.copy()

df_genre = pd.read_csv('movie-genres.csv')
genres = df_genre.copy()

vectorizer= CountVectorizer(
    tokenizer= word_tokenize, 
    token_pattern= None, 
    stop_words= 'english'
)

genres_transformed= vectorizer.fit_transform(genres['definition'])
genres_similarities= cosine_similarity(genres_transformed, genres_transformed)
genres_similarities= pd.DataFrame(genres_similarities, index= genres['genres'], columns= genres['genres'])

def get_recommendations(genre):
    
    # check if the movie is recognized
    if genre not in genres['genres'].values:
        # movie is not recognized, print error message and exit the function
        print(f'Genre {genre} not recognized.\n')
        return

    similarities = genres_similarities
    
    # retrieve the top 3 most similar movies
    top_3_most_similar = similarities.loc[:, genre].nlargest(4).iloc[1:].index.values

    return top_3_most_similar

def get_emotion(emotion):
  genre = ''    
  if(emotion == "Sad"):
      genre = 'Drama'

  elif(emotion == "Disgust"):
      genre = 'Musical'

  elif(emotion == "Anger"):
      genre = 'Family'
  
  elif(emotion == "Anticipation"):
      genre = 'Thriller'
  
  elif(emotion == "Fear"):
      genre = 'Sport'
  
  elif(emotion == "Enjoyment"):
      genre = 'Thriller'
  
  elif(emotion == "Trust"):
      genre = 'Western'

  return genre

form = st.form(key='my_form')
user_mood = form.text_input(label='How are you feeling?')
submit_button = form.form_submit_button(label='Submit')

a = get_emotion(user_mood)
b = get_recommendations(a)
df_genre = movies.loc[movies['genres'] == a][:5]
df_genre_alike_1 = movies.loc[movies['genres'] == b[1]].head(3)
df_genre_alike_2 = movies.loc[movies['genres'] == b[2]].head(2)
results = pd.concat([df_genre, df_genre_alike_1, df_genre_alike_2])
st.markdown(results.style.set_table_styles([dict(selector='*', props=[('text-align', 'center')]), dict(selector='th', props=[('min-width', '150px')])]).to_html(),unsafe_allow_html=True)
