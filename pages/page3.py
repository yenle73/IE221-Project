from library import library as lib
from models import models

movies_filename = "data/movies.csv"
ratings_filename = "data/ratings.csv"

df_movies = lib.pd.read_csv(movies_filename, usecols=['movieId', 'title'], dtype={'movieId': 'int32', 'title': 'str'})
df_ratings = lib.pd.read_csv(ratings_filename, usecols=['userId', 'movieId', 'rating'], dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

### Pre-processing the data ###
df_movies_cnt = lib.pd.DataFrame(df_ratings.groupby('movieId').size(), columns=['count'])
df_movies_cnt['count'].quantile(lib.np.arange(1, 0.6, -0.05))

# Filter Data to take out only Popular Movies
popularity_thres = 50
popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))
df_ratings_drop_movies = df_ratings[df_ratings.movieId.isin(popular_movies)]

df_users_cnt = lib.pd.DataFrame(df_ratings_drop_movies.groupby('userId').size(), columns=['count'])

# Filter for Inactive Users
ratings_thres = 50
active_users = list(set(df_users_cnt.query('count >= @ratings_thres').index))
df_ratings_drop_users = df_ratings_drop_movies[df_ratings_drop_movies.userId.isin(active_users)]

model = models.KNN()
movie_user_mat = model.create_movie_user_matrix(df_ratings_drop_users, 'movieId', 'userId', 'rating')

movie_to_idx = model.create_mapper(df_movies, movie_user_mat)
movie_user_mat_sparse = model.matrix_to_sparse(movie_user_mat)
model_knn = model.knn(movie_user_mat_sparse)

lib.st.markdown("<h2 style='text-align: center; color: #10316B;'>Collaborative Recommender</h2>", unsafe_allow_html=True)
lib.st.markdown("<h5 style='text-align: center; color: #10316B;'>To teach us what you like, please type in a movie that you already know and like.</h5>", unsafe_allow_html=True)

form = lib.st.form(key='my_form')
user_title = form.text_input(label='Enter a movie\'s name')
submit_button = form.form_submit_button(label='Submit')

if submit_button:
    try:
        lib.st.table(model.make_recommendation(model_knn=model_knn, data=movie_user_mat_sparse, fav_movie=user_title, mapper=movie_to_idx, n_recommendations=10))
    except:
        lib.st.warning('Oops! No match is found')