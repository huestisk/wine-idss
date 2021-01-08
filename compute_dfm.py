
import numpy as np
import pandas as pd
import pickle

MAX_IDX = 500

def freq_analysis(tf_idf, verbose=False):

  # make copy
  freq_analysis = tf_idf.copy()

  # compute term frequency
  freq_of_term = np.array([len(doc_ids) for doc_ids in freq_analysis[:,2]])
  term_freq = freq_of_term / sum(freq_of_term)

  # sorted by tf-idf
  ascending_idx = np.argsort(freq_analysis[:,1])
  sorted_tf_idf = freq_analysis[ascending_idx[::-1]]
  sorted_term_freq = term_freq[ascending_idx[::-1]]

  # Frequency analysis
  idx = percentage = 0
  while percentage < 0.25:
    percentage += sorted_term_freq[idx]
    idx += 1

  if verbose:
    print("The " + str(idx+1) + " most common combined words make up more than 25 percent of all words")

  while percentage < 0.5:
    percentage += sorted_term_freq[idx]
    idx += 1

  if verbose:
    print("The " + str(idx+1) + " most common combined words make up more than 50 percent of all words")

  while percentage < 0.75:
    percentage += sorted_term_freq[idx]
    idx += 1

  if verbose:
    print("The " + str(idx+1) + " most common combined words make up more than 75 percent of all words")

  while idx < MAX_IDX - 1:
    percentage += sorted_term_freq[idx]
    idx += 1

  if verbose:
    print("The " + str(MAX_IDX) + " most common combined words make up more than " + str(round(100*percentage,2)) + " percent of all words")

  return sorted_tf_idf[:MAX_IDX]


def convert2dfm(tf_idf, length):

  columns = tf_idf[:,0]
  dfm = np.zeros((length,len(columns)), dtype=float)

  for idx in range(len(tf_idf)):
    for doc in tf_idf[idx,2]:
      dfm[doc, idx] += tf_idf[idx,1]

  return pd.DataFrame(data=dfm, columns=columns)

if __name__=="__main__":

  # load data
  data = pd.read_csv('data/winemag-data-130k-v2.csv')

  # get reviews
  reviews = data['description']
  num_reviews = len(reviews)
  
  try:
    # load tfidf words
    with open('data/tfidf_words.p', 'rb') as fp:
      tfidf_words = pickle.load(fp)
  except:
    raise Exception("The TF-IDF has not been computed")

  # Word frequency analysis for filtering
  final_tf_idf = freq_analysis(tfidf_words)

  # convert to matrix
  dfm = convert2dfm(final_tf_idf, num_reviews)

  # save to pickle
  with open('data/dfm.p', 'wb') as fp:
    pickle.dump(dfm, fp)

  # create training data
  labels = data[{'variety','price','points','province'}].copy()
  
  # save to pickle
  with open('data/labels.p', 'wb') as fp:
    pickle.dump(labels, fp)


  # try:
  #   # load tfidf bigrams
  #   with open('data/tfidf_bigrams.p', 'rb') as fp:
  #     tfidf_bigrams = pickle.load(fp)
  # except:
  #   raise Exception("The TF-IDF has not been computed")


