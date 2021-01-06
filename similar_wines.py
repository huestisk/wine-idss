import numpy as np
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords

from collections import Counter

# download stopwords (common english words)
# nltk.download('stopwords')
stops = set(stopwords.words("english"))

# download tokenizer
# nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def remove_stopwords(words):
  meaningful_words = [w.lower() for w in words if not w.lower() in stops]
  return meaningful_words

def sentence_to_wordlist(raw):
  if len(raw) > 0:
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return remove_stopwords(words)
  else:
    return raw

def get_bigrams(reviews):
  # tokenize
  raw_reviews = [tokenizer.tokenize(review) for review in reviews]

  # get bigrams for each review
  bigrams = np.empty((0,3),dtype=object)
  for review_idx, raw_review in enumerate(raw_reviews):
    for sentence in raw_review:
      # clean & split by word
      clean_sentence = sentence_to_wordlist(sentence)
      # convert to bigram
      for idx, w in enumerate(clean_sentence[:-1]):
        bigram = np.array([[w, clean_sentence[idx+1], review_idx]])
        bigrams = np.append(bigrams, bigram, axis=0)

  return bigrams


def compute_TF_IDF(bigrams, num_reviews):

  output = np.empty((0,3),dtype=object)
  bigram_counts = dict()
  for bigram in bigrams:
    bigram_name = bigram[0] + "_" + bigram[1]
    if bigram_name in bigram_counts:
      bigram_counts[bigram_name].append(bigram[2])
    else:
      output = np.append(output, bigram.reshape(1,3), axis=0)
      bigram_counts[bigram_name] = list(bigram[2])

  num_terms = len(bigrams)
  freq_of_term = np.array([len(value) for value in bigram_counts.values()])
  num_reviews_w_term = np.array([len(set(value)) for value in bigram_counts.values()])

  term_freq = freq_of_term / num_terms
  inverse_document_freq = np.log(num_reviews / num_reviews_w_term)
  tf_idf = term_freq * inverse_document_freq

  output[:,2]=tf_idf

  return output


if __name__=="__main__":

  # load data
  data = pd.read_csv('data/winemag-data-130k-v2.csv')

  # get reviews
  reviews = data['description'][:200]
  
  # bigrams
  bigrams = get_bigrams(reviews)
  
  # compute TF-IDF scores
  num_reviews = len(reviews)
  tf_idf = compute_TF_IDF(bigrams, num_reviews)

  # save to csv
  np.savetxt('output/tf_idf.csv', tf_idf, fmt='%s', delimiter=',')

