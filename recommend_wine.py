import numpy as np
import pandas as pd

import pickle

from sklearn import preprocessing

from preprocessing.preprocessing import preprocess_input
from models.wine_nn import WineNN

ORIGINAL_DATA = "data/data.pkl"
PRICE_RANGE = [0, 100]
TEST_INPUT = "The wine should be an oaky wine from the south of France. Preferably, with an amora of cherry and cat's pee."


class WineRecommender:

	def __init__(self):

		try:
			self.data = pickle.load(open(ORIGINAL_DATA, 'rb'))
		except:
			raise("The dataset is not available.")

		self.variety_model = WineNN("variety")
		self.province_model = WineNN("province")

		self.input_text = None
		self.features = None
		self.price_range = None

	def set_input_text(self, input_text):
		self.features = preprocess_input(input_text)

	def set_price_range(self, price_range):
		self.price_range = price_range

	def recommend(self):
		if self.features is None:
			raise Exception("There is no input text.")
		elif self.price_range is None:
			raise Exception("There is no price range.")

		# Filter
		filters = (self.data["price"] >= self.price_range[0]) & (
			self.data["price"] <= self.price_range[1])
		
		wines = (self.data[filters]).copy()

		# Make predictions
		[varieties, probs_var] = self.variety_model.predict(self.features)
		[provinces, probs_province] = self.province_model.predict(
			self.features)

		# Top classes
		variety = varieties[np.argmax(probs_var)]
		province = provinces[np.argmax(probs_province)]

		# TODO: change to numpy
		provinces = list(provinces)
		varieties = list(varieties)
		probs_var = list(probs_var[0])
		probs_province = list(probs_province[0])

		assert len(probs_var) == len(probs_var)

		# weightsvar * weightsprov * points normalized
		def assign_val(r):
			try:
				probvar = probs_var[varieties.index(str(r['variety']))]
				probprov = probs_province[provinces.index(str(r['province']))]
			except:
				return -1

			p = (r['points'] - 80 + 1) / 21

			return probvar * probprov * (p)

		# Compute ranking
		wines["val"] = wines.apply(assign_val, axis=1)

		# Sort
		sorted_wines = wines.nlargest(10, 'val')

		return variety, province, sorted_wines[{"title", "price", "variety", "country", "province"}]


if __name__ == "__main__":

	wine_recommender = WineRecommender()
	wine_recommender.set_input_text(TEST_INPUT)
	wine_recommender.set_price_range(PRICE_RANGE)
	variety, province, wines = wine_recommender.recommend()

	print(wines.head())
