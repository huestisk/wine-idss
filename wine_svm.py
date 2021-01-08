"""Train SVM"""

import numpy as np
import pickle

class WineSVM:
  pass


if __name__=="__main__":
  
  try:
    # load tfidf words
    with open('data/tfidf_words.p', 'rb') as fp:
      tfidf_words = pickle.load(fp)

    # load tfidf bigrams
    with open('data/tfidf_bigrams.p', 'rb') as fp:
      tfidf_bigrams = pickle.load(fp)

  except:
    raise Exception("The TF-IDF has not been computed")

  # find most important
  freq_of_term = np.array([len(doc_ids) for doc_ids in tfidf_words[:,2]])
  num_reviews_w_term = np.array([len(set(doc_ids)) for doc_ids in tfidf_words[:,2]])

  term_freq = freq_of_term / sum(freq_of_term)

  # convert to matrix
