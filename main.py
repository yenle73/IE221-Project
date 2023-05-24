import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import time
from PIL import Image


st.markdown("<h1 style='text-align: center; color: #3F3F3F;'>Movie Recommendation System</h1>", unsafe_allow_html=True)
image = Image.open(r'D:\IE221-Project\bg.png')
st.image(image, width=650)

form = st.form(key='my_form')
user_title = form.text_input(label='Enter a movie')
submit_button = form.form_submit_button(label='Submit')

# Load dataset
df = pd.read_csv(r'D:\IE221-Project\final_data.csv')
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
    return_df['Similariry Score'] = [sim_scores[i][1] for i in range(10)]
    return return_df

all_titles = [df2['title'][i] for i in range(len(df2['title']))]
user_title = user_title.lower()

if user_title in all_titles:
    with st.spinner('Searching for movies...'):
        time.sleep(3)
        results = get_recommendations(user_title)

    st.success('Matches Found!', icon="✅")
    st.markdown(f"<h2 style='text-align: center; color: #3F3F3F;'>TOP 10 Movies Similar to \"{user_title}\"</h2>", unsafe_allow_html=True)
    st.markdown(results.style.set_table_styles([dict(selector='*', props=[('text-align', 'center')]), dict(selector='th', props=[('min-width', '150px')])]).to_html(),unsafe_allow_html=True)
else:
    st.warning('Movie Not Found! Please Try Again!')

