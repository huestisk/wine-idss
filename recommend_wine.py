import numpy as np
import pandas as pd

import re
import pickle

import nltk
from nltk.corpus import stopwords
from combine_by_stem import replace_by_common_descriptors


from sentence_transformers import SentenceTransformer, util

from wine_nn import WineNN

# nltk.download('stopwords') # TODO: uncomment if not downloaded
stops = set(stopwords.words("english"))

# nltk.download('punkt') # TODO: uncomment if not downloaded
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

TEST = "The wine should be an oaky white wine from the south of France. Preferably, with an amora of cherry and cat's pee."
MAX_PRICE = 100

class WineRecommender:

  def __init__(self, data):
    self.data = data#.dropna(subset=['variety' , 'province', 'points', 'price'])

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

    [varieties, probs_var] = self.variety_model.predict(self.features)
    [provinces, probs_province] = self.province_model.predict(self.features)
    provinces=list(provinces)
    varieties=list(varieties)
    probs_var=list(probs_var[0])
    probs_province=list(probs_province[0])
    # Assume that we return list of probabilities
    # weightsvar * weightsprov * sim * poits normalized (filtered by price)
    def assign_val(r):
      assert len(probs_var) == len(probs_var)
      try:
        ind_var = varieties.index(str(r['variety'])) # np.where(varieties == r['variety'])
        probvar = probs_var[ind_var]
        ind_prov = provinces.index(str(r['province'])) #np.where(provinces == r['province'])
        probprov = probs_province[ind_prov]
      except:
        print(f"pb with finding {r['variety']} or {r['province']}")
        return -1
      p = (r['points'] - 80 + 1) /21
      
      if r['price'] > self.max_price:
        return 0
      # compute sims values
      
      val = probvar*probprov * (p)
      print(f"{val=} for {r['variety']} or {r['province']}")
      return val
    

    # print("\nThe predicted variety is " + variety + " and the predicted province is " + province + ".\n")

    # filter dataframe
    # filters = (self.data["variety"] == variety) & \
    #           (self.data["province"] == province) & (self.data["price"] < self.max_price)

    # filtered_wines = self.data.loc[filters]

    # # sort
    # filtered_wines = filtered_wines.sort_values(by='points')

    # # return filtered_wines['title'].iloc[0]
    self.data["val"] = self.data.apply(assign_val, axis=1)
    vc = self.data["val"].value_counts()
    missings = vc[-1]/len(self.data["val"])
    print(missings)
    filtered_wines = self.data.sort_values(by='val')
    return filtered_wines[{"title", "price", "variety", "country", "province"}].iloc[:10]

    


if __name__=="__main__":

  # load data
  data = pd.read_csv('data/winemag-data-130k-v2.csv')
  
  wine_recommender = WineRecommender(data)

  wine_recommender.set_input_text(TEST)

  wine_recommender.set_max_price(MAX_PRICE)

  wines = wine_recommender.recommend()

  print("You should try the wines " + wines['title'])


