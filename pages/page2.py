from library import library as lib
from models import models

df1 = lib.pd.read_csv('data/final_data.csv')
df_movies = df1.copy()

df2 = lib.pd.read_csv('data/movie-genres.csv')
df_genres = df2.copy()

model = models.similarity_model()
genres_similarities = model.similarity(df_genres, 'definition')
genres_similarities= lib.pd.DataFrame(genres_similarities, index= df_genres['genres'], columns= df_genres['genres'])

def get_genre_recommendations(genre):
    if genre not in df_genres['genres'].values:
        # movie is not recognized, print error message and exit the function
        lib.st.warning(f'Genre {genre} not recognized.\n')
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
      genre = 'Thriller'

  elif(emotion == "Anger"):
      genre = 'Action'
  
  elif(emotion == "Anticipation"):
      genre = 'Thriller'
  
  elif(emotion == "Fear"):
      genre = 'Thriller'
  
  elif(emotion == "Enjoyment"):
      genre = 'Science Fiction'
  
  elif(emotion == "Trust"):
      genre = 'Family'

  return genre

lib.st.markdown("<h2 style='text-align: center; color: #10316B;'>Mood and Genre Based Recommnder</h2>", unsafe_allow_html=True)
lib.st.markdown("<h5 style='text-align: center; color: #10316B;'>Please select your mood below for us to make recommendations.</h5>", unsafe_allow_html=True)

form = lib.st.form("my_form")
option = form.selectbox(
    'How do you wish to feel?',
    ("Sad", "Disgust", "Anger", "Anticipation", "Fear", "Enjoyment", "Trust"))
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
    b  = get_genre_recommendations(a)
    df_genre = df_movies.loc[df_movies['genres'] == a].sample(n=5)
    df_genre_alike_1 = df_movies.loc[df_movies['genres'] == b[1]].sample(n=3)
    df_genre_alike_2 = df_movies.loc[df_movies['genres'] == b[2]].sample(n=2)
    results = lib.pd.concat([df_genre, df_genre_alike_1, df_genre_alike_2])
    results = results.drop('overview', axis=1)
    results['release_year'] = results['release_year'].apply(str)
    results.columns = ['Title', 'Genre', 'Release Year']
    lib.st.success('Matches Found!')
    lib.st.markdown(f"<h3 style='text-align: center; color: #10316B;'>Movies You May Like</h3>", unsafe_allow_html=True)
    #st.markdown(results.style.set_table_styles([dict(selector='*', props=[('text-align', 'center')]), dict(selector='th', props=[('min-width', '150px')])]).to_html(),unsafe_allow_html=True)
    lib.st.table(results)
