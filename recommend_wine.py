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

INPUT_SIZE = 500
TEST = "The wine should be an oaky white wine from the south of France. Preferably, with an amora of cherry and cat's pee."

class WineRecommender:

  def __init__(self, input_text):
    self.input_text = self.convert_text(input_text)
    self.model = WineNN()

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
      with open('output/tfidf_dict.p', 'rb') as fp:
        tf_idf = pickle.load(fp)

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

  def predict(self, features):
    return self.model.predict(features)
    

if __name__=="__main__":
  
  wine_recommender = WineRecommender(TEST)

  print(wine_recommender.input_text)

  features = wine_recommender.get_features()

  prediction = wine_recommender.predict(features)

  print("\nThe predicted variety is " + prediction)

