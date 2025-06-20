from data import ratings,movies
#Trending now
import pandas as pd
from datetime import datetime, timedelta

ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
cut_off = datetime.now() - timedelta(days=30)
recent_ratings = ratings[ratings['timestamp']>= cut_off]

group_movies = movies.groupby(['movieId','genres','title'])
movie_counts = group_movies.size()

average = recent_ratings['rating'].mean()
rating_count = recent_ratings['rating'].count()

rating_count = recent_ratings.groupby('movieId')['rating'].count()
top_ten = rating_count.sort_values(ascending=False).head(10)

print(top_ten)