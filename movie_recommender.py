import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Loading in the dataset 
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings1.csv")

# Creating a matrix
matrix  = ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating')
matrix.fillna(0,inplace = True)
print(matrix.head())

# Reduce noise by adding filters to the dataset 

no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

# Visualizing movies that have more than 10 votes 
f,ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(no_user_voted.index,no_user_voted,color='mediumseagreen')
plt.axhline(y=10,color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()

# Adding the > 10 threshold for movies 
matrix = matrix.loc[no_user_voted[no_user_voted > 10].index,:]

# Visualizing users that have voted for more than 50 movies 
f,ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(no_movies_voted.index,no_movies_voted,color='mediumseagreen')
plt.axhline(y=50,color='r')
plt.xlabel('UserId')
plt.ylabel('No. of votes by user')
plt.show()

# Adding the > 50 threshold for users 
matrix=matrix.loc[:,no_movies_voted[no_movies_voted > 50].index]
matrix

# Applying csr matrix function to dataset to remove sparseness
csr_data = csr_matrix(matrix.values)
matrix.reset_index(inplace=True)

# Creating the ML algorithm 

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Defining function that provides 10 movies, sorted by the closest cosine distance 
def recommend_movie(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = matrix[matrix['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = matrix.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    else:
        return "No movies found. Please check your input"
    
print(recommend_movie('Whiplash'))


