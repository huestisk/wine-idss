import numpy as np
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords

from combine_by_stem import replace_by_common_descriptors

import pickle

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

def get_sentences(reviews):
  # tokenize
  raw_reviews = [tokenizer.tokenize(review) for review in reviews]
  # convert
  sentences = []
  for raw_review in raw_reviews:
    for sentence in raw_review:
      # clean & split by word
      clean_sentence = sentence_to_wordlist(sentence)
      # replace by common discriptor
      replaced_sentence = replace_by_common_descriptors(clean_sentence)
      # append
      sentences.append(replaced_sentence)

  return sentences


def get_bigrams(reviews):
  # tokenize
  raw_reviews = [tokenizer.tokenize(review) for review in reviews]
  # get bigrams for each review
  bigrams = np.empty((0,3),dtype=object)
  for review_idx, raw_review in enumerate(raw_reviews):
    for sentence in raw_review:
      # clean & split by word
      clean_sentence = sentence_to_wordlist(sentence)
      # replace by common discriptor
      replaced_sentence = replace_by_common_descriptors(clean_sentence)
      # convert to bigram
      for idx, w in enumerate(replaced_sentence[:-1]):
        bigram = np.array([[w, replaced_sentence[idx+1], review_idx]])
        bigrams = np.append(bigrams, bigram, axis=0)

  return bigrams


def compute_TF_IDF_bigrams(bigrams, num_reviews):
  """Computes the TF-IDF values"""

  output = np.empty((0,4),dtype=object)
  
  bigram_idxs = dict()
  for bigram in bigrams:
    bigram_name = bigram[0] + "_" + bigram[1]
    if bigram_name in bigram_idxs:
      bigram_idxs[bigram_name].append(bigram[2])
    else:
      tmp = np.array([bigram[0], bigram[1], np.newaxis, np.newaxis])
      output = np.append(output, tmp.reshape(1,4), axis=0)
      bigram_idxs[bigram_name] = list(bigram[2])

  num_terms = len(bigrams)
  freq_of_term = np.array([len(value) for value in bigram_idxs.values()])
  num_reviews_w_term = np.array([len(set(value)) for value in bigram_idxs.values()])

  term_freq = freq_of_term / num_terms
  inverse_document_freq = np.log(num_reviews / num_reviews_w_term)
  tf_idf = term_freq * inverse_document_freq

  output[:,2] = np.array(list(bigram_idxs.values()),dtype=object)
  output[:,3] = tf_idf

  return output


if __name__=="__main__":

  # load data
  data = pd.read_csv('data/winemag-data-130k-v2.csv')

  # get reviews
  reviews = data['description']

  # get sentences
  sentences = get_sentences(reviews)

  # save
  with open('data/sentences.p', 'wb') as fp:
    pickle.dump(sentences, fp)
  
  # # bigrams
  # bigrams = get_bigrams(reviews)
  
  # # compute TF-IDF scores
  # num_reviews = len(reviews)
  # tf_idf = compute_TF_IDF_bigrams(bigrams, num_reviews)

  # # save to csv
  # np.savetxt('output/tf_idf.csv', tf_idf, fmt='%s', delimiter=';')


