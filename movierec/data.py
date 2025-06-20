import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
movies= pd.read_csv('ml-latest-small/movies.csv', names=['movieId','title','genres'])
ratings = pd.read_csv('ml-latest-small/ratings.csv')
#rec_ratings = pd.read_csv('ml-latest/ratings.csv', usecols=['timestamp'])
#nandf = df.isnull()
#print(nandf.to_string())
ratings_matrix = ratings.pivot_table(
    index = 'userId',
    columns = 'movieId',
    values = 'rating',
    fill_value = 0
)
ratings_matrix = ratings_matrix.astype('float32')
sparse_matrix = csr_matrix(ratings_matrix).toarray()
print(sparse_matrix)
U, S, VT = np.linalg.svd(sparse_matrix)
print("U shape =", U.shape)
print("Singular values (S) =", S)
print("V^T shape =", VT.shape)


