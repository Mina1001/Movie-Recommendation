from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import pandas as pd

ratings = pd.read_csv('ml-latest-small/ratings.csv')
reader = Reader(rating_scale=(0.5, 5.0))

dataset = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)

trainset,testset = train_test_split(dataset,   test_size=0.25)
model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
model.fit(trainset)
predictions = model.test(testset)

accuracy.rmse(predictions)
accuracy.mae(predictions)
cross_validate(model, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)