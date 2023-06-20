from library import library as lib

lib.st.markdown("<h1 style='text-align: center; color: #10316B;'>Movie Recommendation System</h1> <h3 style='text-align: center; color: #10316B;'>&#10024 Discover New Movies &#10024</h3> <h5 style='text-align: center; color: #EBB02D;'>By Le Thi Kim Yen</h5> <h5 style='text-align: center; color: #EBB02D;'>Student ID: 21521695</h5>", unsafe_allow_html=True)
image = lib.Image.open('bg.png')
placeholder= lib.st.image(image)

lib.st.sidebar.header("Select movie information")

form = lib.st.sidebar.form("my_form")
option = form.selectbox(
    'Choose a genre',
    ("Action", "Adventure", "Fantasy", "Animation", "Science Fiction", 
     "Drama", "Thriller", "Family", "Comedy", "History", "War", "Western",
     "Romance", "Crime", "Mystery", "Horror", "Documentary", "Music", "TV Movie", "Foreign"))
year = form.slider("Choose release year", 1916, 2023, 2020, 1)
submitted = form.form_submit_button("Submit")

df1 = lib.pd.read_csv('data/final_data.csv')
df_movies = df1.copy()

if submitted:
    placeholder.empty()
    results = df_movies[df_movies['genres'] == option]
    results = results[results['release_year'] == year]
    results.columns = ["Title", "Overview", "Genre", "Release Year"]
    if results.shape[0] == 0:
        lib.st.warning(f"There's no {option} movie released in {year} exists in database.")
    elif results.shape[0] < 10:
        results = results
        lib.st.dataframe(results)
    else: 
        results = results.sample(n = 10)
        lib.st.dataframe(results)