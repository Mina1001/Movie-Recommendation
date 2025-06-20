from data import ratings
from train import model
def recommend_movies(user_id, n=10):

    all_movies = ratings['movieId'].unique()

    seen_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()

    predictions = [model.predict(user_id, movie_id) for movie_id in all_movies if movie_id not in seen_movies]
    

    predictions.sort(key=lambda x: x.est, reverse=True)

    top_n = predictions[:n]
    return [(pred.iid, pred.est) for pred in top_n]

recommendations = recommend_movies(1, n=5)
#print()