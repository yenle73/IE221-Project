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

df1 = pd.read_csv('data/final_data.csv')
df_movies = df1.copy()

df2 = pd.read_csv('data/movie-genres.csv')
df_genres = df2.copy()

vectorizer= CountVectorizer(
    tokenizer= word_tokenize, 
    token_pattern= None, 
    stop_words= 'english'
)

genres_transformed= vectorizer.fit_transform(df_genres['definition'])
genres_similarities= cosine_similarity(genres_transformed, genres_transformed)
genres_similarities= pd.DataFrame(genres_similarities, index= df_genres['genres'], columns= df_genres['genres'])

def get_recommendations(genre):
    
    # check if the movie is recognized
    if genre not in df_genres['genres'].values:
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
      genre = 'Thriller'
  
  elif(emotion == "Enjoyment"):
      genre = 'Science Fiction'
  
  elif(emotion == "Trust"):
      genre = 'Crime'

  return genre

st.markdown("<h2 style='text-align: center; color: #10316B;'>Mood and Genre Based Recommnder</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #10316B;'>Recommendations will be made based on what you're feeling, please select your mood below.</h5>", unsafe_allow_html=True)

form = st.form("my_form")
option = form.selectbox(
    'How are you feeling?',
    ("Sad", "Digust", "Anger", "Anticipation", "Fear", "Enjoyment", "Trust"))
submitted = form.form_submit_button("Submit")

mood = ''

if option == "Sad":
    mood = "Sad"
elif option == "Disgust":
    mood = "Disgust"
elif option == "Anger":
    mood = "Anger"
elif option == "Anticipation":
    mood = "Anticipation"
elif option == "Fear":
    mood = "Fear"
elif option == "Enjoyment":
    mood = "Enjoyment"
elif option == "Trust":
    mood = "Trust"

if submitted:
    a = get_emotion(mood)
    b  = get_recommendations(a)
    df_genre = df_movies.loc[df_movies['genres'] == a][:5]
    df_genre_alike_1 = df_movies.loc[df_movies['genres'] == b[1]].head(3)
    df_genre_alike_2 = df_movies.loc[df_movies['genres'] == b[2]].head(2)
    results = pd.concat([df_genre, df_genre_alike_1, df_genre_alike_2])
    results = results.drop('overview', axis=1)
    #st.markdown(results.style.set_table_styles([dict(selector='*', props=[('text-align', 'center')]), dict(selector='th', props=[('min-width', '150px')])]).to_html(),unsafe_allow_html=True)
    st.table(results)
