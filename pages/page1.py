import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import time
from PIL import Image
import random

# Load dataset
df = pd.read_csv('data/final_data.csv')
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
    sim_scores = sim_scores[1:21]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # return list of similar movies
    return_df = pd.DataFrame(columns=['Title','Release Year'])
    return_df['Title'] = df['title'].iloc[movie_indices]
    return_df['Overview'] = df['overview'].iloc[movie_indices].str[:500]
    return_df['Release Year'] = df2['release_year'].iloc[movie_indices]
    return_df['Similarity Score'] = [sim_scores[i][1] for i in range(20)]
    return_df = return_df.drop('Similarity Score', axis=1)
    #random.shuffle(return_df)
    return return_df

all_titles = [df2['title'][i] for i in range(len(df2['title']))]


st.markdown("<h2 style='text-align: center; color: #10316B;'>Content Based Recommnder</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #10316B;'>To teach us what you like, please type in 3 movies that you already know and like.</h5>", unsafe_allow_html=True)

#form_1 = st.form(key='my_form_1')
#user_title = form_1.text_input(label='Enter a movie\'s name')
#submit_button = form_1.form_submit_button(label='Submit')

form_1 = st.form(key='my_form')
user_title_1 = form_1.text_input(label='Enter a movie\'s name', key='ut1')
#user_title_2 = form_1.text_input(label='Enter a movie\'s name', key='ut2')
#user_title_3 = form_1.text_input(label='Enter a movie\'s name', key='ut3')
submit_button = form_1.form_submit_button(label='Submit')


user_title_1 = user_title_1.lower()
#user_title_2 = user_title_2.lower()
#user_title_3 = user_title_3.lower()


if submit_button:
    if user_title_1 in all_titles:
        with st.spinner('Searching for movies...'):
            time.sleep(3)
            results_1 = get_recommendations(user_title_1)
            #results_2 = get_recommendations(user_title_2)
            #results_3 = get_recommendations(user_title_3)
            #dfs = [results_1, results_2, results_3]
            #results = pd.concat(dfs, axis=0)

        st.success('Matches Found!')
        st.markdown(f"<h3 style='text-align: center; color: #10316B;'>Movies You May Like</h3>", unsafe_allow_html=True)
        #st.markdown(results.style.set_table_styles([dict(selector='*', props=[('text-align', 'center')]), dict(selector='col', props=[('max-width', 800)])]).to_html(),unsafe_allow_html=True)
        st.dataframe(results_1[:10])
        if st.button('Show more'):
            st.dataframe(results_1[11:20])
        else: st.stop()
    else:
        st.warning('Movie Not Found! Please Try Again!')
        st.stop()
