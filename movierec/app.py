from flask import Flask, jsonify
import requests
from train import model
from data import ratings

app = Flask(__name__)

def recommend_movies(user_id, n=5):
    all_movies = ratings['movieId'].unique()
    seen_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    predictions = [model.predict(user_id, movie_id) for movie_id in all_movies if movie_id not in seen_movies]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n]
    return [{'movieId': pred.iid, 'predicted_rating': pred.est} for pred in top_n]

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        user_id = int(requests.args.get('user_id'))
        n = int(requests.args.get('n', 5))
        recommendations = recommend_movies(user_id, n)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
