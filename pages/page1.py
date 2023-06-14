from library import library as lib
from models import models

# Load dataset
df = lib.pd.read_csv('data/final_data.csv')
df2 = df.copy()

df2['title'] = df2['title'].str.lower()
df2['release_year'] = df2['release_year'].apply(str)
df2 = df2.reset_index()
indices = lib.pd.Series(df2.index, index=df2['title']).drop_duplicates()

model = models.similarity_model()
cosine_sim = model.similarity(df2, 'overview')
all_titles = [df2['title'][i] for i in range(len(df2['title']))]

lib.st.markdown("<h2 style='text-align: center; color: #10316B;'>Content Based Recommnder</h2>", unsafe_allow_html=True)
lib.st.markdown("<h5 style='text-align: center; color: #10316B;'>To teach us what you like, please type in 3 movies that you already know and like.</h5>", unsafe_allow_html=True)

#form_1 = st.form(key='my_form_1')
#user_title = form_1.text_input(label='Enter a movie\'s name')
#submit_button = form_1.form_submit_button(label='Submit')

form_1 = lib.st.form(key='my_form')
user_title_1 = form_1.text_input(label='Enter a movie\'s name', key='ut1')
user_title_2 = form_1.text_input(label='Enter a movie\'s name', key='ut2')
user_title_3 = form_1.text_input(label='Enter a movie\'s name', key='ut3')
submit_button = form_1.form_submit_button(label='Submit')


user_title_1 = user_title_1.lower()
user_title_2 = user_title_2.lower()
user_title_3 = user_title_3.lower()

if submit_button:
    if user_title_1 in all_titles:
        results_1 = model.get_recommendations(df2, user_title_1, indices, cosine_sim)
        results_2 = model.get_recommendations(df2, user_title_2, indices, cosine_sim)
        results_3 = model.get_recommendations(df2, user_title_3, indices, cosine_sim)
        dfs = [results_1, results_2, results_3]
        results = lib.pd.concat(dfs, axis=0)
        results.sort_values(by=['Similarity Score'])

        lib.st.success('Matches Found!')
        lib.st.markdown(f"<h3 style='text-align: center; color: #10316B;'>Movies You May Like</h3>", unsafe_allow_html=True)
        #st.markdown(results.style.set_table_styles([dict(selector='*', props=[('text-align', 'center')]), dict(selector='col', props=[('max-width', 800)])]).to_html(),unsafe_allow_html=True)
        lib.st.dataframe(results[:10])
    else:
        lib.st.warning('Movie Not Found! Please Try Again!')
