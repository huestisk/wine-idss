import numpy as np
import pandas as pd

import re
import pickle

import nltk
from nltk.corpus import stopwords
from combine_by_stem import replace_by_common_descriptors

from wine_nn import WineNN

# nltk.download('stopwords') # TODO: uncomment if not downloaded
stops = set(stopwords.words("english"))

# nltk.download('punkt') # TODO: uncomment if not downloaded
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

TEST = "The wine should be an oaky white wine from the south of France. Preferably, with an amora of cherry and cat's pee."
MAX_PRICE = 100

class WineRecommender:

  def __init__(self, data):

    self.data = data

    self.variety_model = WineNN("variety")
    self.province_model = WineNN("province")

    self.input_text = None    
    self.features = None
    self.max_price = None

  def set_input_text(self, input_text):
    self.input_text = self.convert_text(input_text)
    self.features = self.get_features()

  def set_max_price(self, max_price):
    self.max_price = max_price

  def clean_sentences(self, sentence):
    cleaned_sentence = []
    if len(sentence) > 0:
      words = re.sub("[^a-zA-Z]"," ", sentence).split()
      cleaned_sentence = [w.lower() for w in words if not w.lower() in stops]

    return cleaned_sentence

  def convert_text(self, input_text):
    # Tokenize
    sentences = tokenizer.tokenize(input_text)

    # Clean
    clean_sentences = [self.clean_sentences(sentence) for sentence in sentences]

    # Replace Words
    replaced_sentences =  [replace_by_common_descriptors(clean_sentence) for clean_sentence in clean_sentences]

    # Flatten
    return [word for sentence in replaced_sentences for word in sentence]


  def get_features(self):
  
    # load dictionaries
    try:
      # with open('output/tfidf_dict.p', 'rb') as fp:
      #   tf_idf = pickle.load(fp)

      with open('output/feature_dict.p', 'rb') as fp:
        feature_dict = pickle.load(fp)

    except:
      raise Exception("Could not load files.")

    # compute features
    features = np.zeros(len(feature_dict))
    for word in self.input_text:
      if word in feature_dict:
        idx = feature_dict[word]
        # Add tf_idf value
        # features[idx] += tf_idf[word]
        # Alternative: count
        features[idx] += 1

    return features
    
  def recommend(self):
    if self.features is None:
      raise Exception("There is not input text.")

    variety = self.variety_model.predict(self.features)
    province = self.province_model.predict(self.features)

    # print("\nThe predicted variety is " + variety + " and the predicted province is " + province + ".\n")

    # filter dataframe
    filters = (self.data["variety"] == variety) & \
              (self.data["province"] == province) & (self.data["price"] < self.max_price)

    filtered_wines = self.data.loc[filters]

    # sort
    filtered_wines = filtered_wines.sort_values(by='points')

    # return filtered_wines['title'].iloc[0]
    return filtered_wines[{"title", "price", "variety", "country", "province"}]


if __name__=="__main__":

  # load data
  data = pd.read_csv('data/winemag-data-130k-v2.csv')
  
  wine_recommender = WineRecommender(data)

  wine_recommender.set_input_text(TEST)

  wine_recommender.set_max_price(MAX_PRICE)

  wines = wine_recommender.recommend()

  print("You should try the wine " + wines['title'].iloc[0])


