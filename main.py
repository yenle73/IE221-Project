import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import time
from PIL import Image

# Load dataset
df = pd.read_csv('final_data.csv')
df2 = df.copy()

# Drop NA
df2['overview'].fillna(' ', inplace=True)
df2['title'] = df2['title'].str.lower()

# Constructing TF-IDF Matrix
tfidfv=TfidfVectorizer(analyzer='word', stop_words='english')
tfidfv_matrix=tfidfv.fit_transform(df2['overview'])

# Computing Similarity Score
cosine_sim = linear_kernel(tfidfv_matrix, tfidfv_matrix)

df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

df2['release_year'] = df2['release_year'].apply(str)

def get_recommendations(title):
    global sim_scores
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # return list of similar movies
    return_df = pd.DataFrame(columns=['Title','Release Year'])
    return_df['Title'] = df['title'].iloc[movie_indices]
    return_df['Release Year'] = df2['release_year'].iloc[movie_indices]
    return_df['Similarity Score'] = [sim_scores[i][1] for i in range(10)]
    return return_df

st.markdown("<h1 style='text-align: center; color: #10316B;'>Movie Recommendation System</h21>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #EBB02D;'>By Yen Le</h4>", unsafe_allow_html=True)
image = Image.open('bg.png')
st.image(image, width=700)

form = st.form(key='my_form')
user_title = form.text_input(label='Enter a movie')
submit_button = form.form_submit_button(label='Submit')

all_titles = [df2['title'][i] for i in range(len(df2['title']))]
user_title = user_title.lower()

if submit_button:
    if user_title in all_titles:
        with st.spinner('Searching for movies...'):
            time.sleep(3)
            results = get_recommendations(user_title)

        st.success('Matches Found!')
        st.markdown(f"<h3 style='text-align: center; color: #10316B;'>TOP 10 Movies Similar to \"{user_title}\"</h3>", unsafe_allow_html=True)
        st.markdown(results.style.set_table_styles([dict(selector='*', props=[('text-align', 'center')]), dict(selector='th', props=[('min-width', '150px')])]).to_html(),unsafe_allow_html=True)
    else:
        st.warning('Movie Not Found! Please Try Again!')

