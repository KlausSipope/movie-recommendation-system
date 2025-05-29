import pandas as pd
import streamlit as st
import pickle
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

def hybrid(userId, title):
    idx = indices.loc[title]
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = sdf.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: algo.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies['title'].head(5)

ratings = pd.read_csv('ratings_small.csv')

id_map = pd.read_pickle('tmdb_to_movielens_map.pkl')
indices = pd.read_pickle('title_indices.pkl')
sdf = pd.read_pickle('movie_metadata.pkl')
indices_map = pd.read_pickle('id_to_movieid_map.pkl')
cosine_sim = pickle.load(open('cosine_similarity_matrix.pkl', 'rb'))

#movie_title = pickle.load(open('title.pkl', 'rb'))
#algo = SVD(n_factors = 200 , lr_all = 0.005 , reg_all = 0.02 , n_epochs = 40 , init_std_dev = 0.05)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
algo = SVD(n_factors = 200 , lr_all = 0.005 , reg_all = 0.02 , n_epochs = 40 , init_std_dev = 0.05)
trainset = data.build_full_trainset()
algo.fit(trainset)

user_id = sdf['id']
movie_title = sdf['title']

st.title('Movie Recommendation System')

selected_movie_title = st.selectbox(
    'Movie Title',
    movie_title
)	

selected_user_id = st.selectbox(
    'User ID',
    user_id
)	

# Convert selected_movie_title to string and selected_user_id to int
selected_movie_title_str = str(selected_movie_title)
selected_user_id_int = int(selected_user_id)

if st.button('Show Recommendation'):
    recommendations = hybrid(selected_user_id_int, selected_movie_title_str)
    for i in recommendations:
        st.write(i)
