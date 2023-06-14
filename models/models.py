from library import library as lib

class similarity_model:

    def __init__(self):
        pass

    def similarity(self, df, col):
        '''Hàm chuyển đổi văn bản thành dạng vector, tính toán trọng số tf-idf 
        và tính độ tương đồng cho các vector trong ma trận trọng số.
        
        input: dataframe và cột cần biến đổi
        output 
        '''
        
        # Constructing TF-IDF Matrix
        tfidfv = lib.TfidfVectorizer(analyzer='word', stop_words='english')
        tfidfv_matrix = tfidfv.fit_transform(df[col])

        # Computing Similarity Score
        cosine_sim = lib.linear_kernel(tfidfv_matrix, tfidfv_matrix)

        return cosine_sim
    
    def get_recommendations(self, df, title, indices, cosine_sim):
        global sim_scores
        # Get the index of the movie that matches the title
        idx = indices[title]
        # Get the pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:22]
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # return list of similar movies
        return_df = lib.pd.DataFrame(columns=['Title','Release Year'])
        return_df['Title'] = df['title'].iloc[movie_indices]
        return_df['Overview'] = df['overview'].iloc[movie_indices].str[:500]
        return_df['Release Year'] = df['release_year'].iloc[movie_indices]
        return_df['Similarity Score'] = [sim_scores[i][1] for i in range(21)]
        #return_df = return_df.drop('Similarity Score', axis=1)
        #random.shuffle(return_df)
        return return_df
    
class KNN():
    def __init__(self):
        pass
    def create_movie_user_matrix(self, df, index, cols, value):
        # Pivot and create movie-user matrix
        movie_user_mat = df.pivot(index=index, columns=cols, values=value).fillna(0)
        return movie_user_mat
    
    def create_mapper(self, df_movies, movie_user_mat):
        # create mapper from movie title to index
        movie_to_idx = {
        movie: i for i, movie in 
        enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title))
    }
        return movie_to_idx
    
    def matrix_to_sparse(self, movie_user_mat):
    # transform matrix to scipy sparse matrix
        movie_user_mat_sparse = lib.csr_matrix(movie_user_mat.values)
        return movie_user_mat_sparse
    
    def knn(self, movie_user_mat_sparse):
        model_knn = lib.NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        model_knn.fit(movie_user_mat_sparse)
        return model_knn

    def fuzzy_matching(self, mapper, fav_movie, verbose=True):
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
            ratio = lib.fuzz.ratio(title.lower(), fav_movie.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            lib.st.warning('Oops! No match is found')
            return
        if verbose:
            lib.st.write('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
        return match_tuple[0][1]
    
    def make_recommendation(self, model_knn, data, mapper, fav_movie, n_recommendations):
        model_knn.fit(data)
        #the function below is a helper function defined to check presence of Movie Name
        idx = self.fuzzy_matching(mapper, fav_movie, verbose=True)
        distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
        # get list of raw idx of recommendations
        raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        # get reverse mapper
        reverse_mapper = {v: k for k, v in mapper.items()}
        # print recommendations
        lib.st.markdown(f"<h3 style='text-align: center; color: #10316B;'>People Who Likes \"{fav_movie}\" Also Likes</h3>", unsafe_allow_html=True)
        df = lib.pd.DataFrame(columns = ['Title'])
        for i, (idx, dist) in enumerate(raw_recommends):
        #st.markdown(f"<p >{i+1}. {reverse_mapper[idx]}</p>", unsafe_allow_html=True)
        #results[i+1] = reverse_mapper[idx]
            df.loc[i+1] = reverse_mapper[idx]

        return df
    
    
    




