import numpy as np
import pandas as pd

import re
import pickle
import nltk
from nltk.corpus import stopwords

from combine_by_stem import replace_by_common_descriptors
from combine_by_vec import Wine2Vec


# download stopwords (common english words)
# nltk.download('stopwords') # TODO: uncomment if not downloaded
stops = set(stopwords.words("english"))

# download tokenizer
# nltk.download('punkt') # TODO: uncomment if not downloaded
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


def preprocess_reviews(reviews, wine2vec):
  # load conversion map
  try:
    with open ('data/conversion_map.p', 'rb') as fp:
      conversion_map = pickle.load(fp)
  except:
    raise Exception("Conversion map not available.")

  # tokenize
  raw_reviews = [tokenizer.tokenize(review) for review in reviews]

  # get words and bigrams for each review
  words = dict()
  bigrams = dict()
  for review_idx, raw_review in enumerate(raw_reviews):
    if review_idx % 10000 == 0:
      print("Getting words and bigrams is " + str(round(100 * review_idx / len(raw_reviews))) + " percent complete.")

    for sentence in raw_review:
      # clean & split by word
      clean_sentence = sentence_to_wordlist(sentence)

      # replace by common discriptor
      replaced_sentence = replace_by_common_descriptors(clean_sentence)

      # combine by wine2vec model
      combined_sentence = [conversion_map[word] if word in conversion_map else word for word in replaced_sentence]

      # add words to dict
      for word in combined_sentence:
        if word in words:
          words[word].append(review_idx)
        else:
          words[word] = [review_idx]

      # convert to bigram TODO: bigrams should consider if stopwords were removed
      for idx, w in enumerate(combined_sentence[:-1]):
        bigram = w  + "_" + combined_sentence[idx+1]
        if bigram in bigrams:
          bigrams[bigram].append(review_idx)
        else:
          bigrams[bigram] = [review_idx]

  print("Getting words and bigrams is complete.")
  return words, bigrams

def compute_TF_IDF(words, num_reviews):
  """Computes the TF-IDF values"""

  freq_of_term = np.array([len(value) for value in words.values()])
  num_reviews_w_term = np.array([len(set(value)) for value in words.values()])

  term_freq = freq_of_term / sum(freq_of_term)
  inverse_document_freq = np.log(num_reviews / num_reviews_w_term)
  tf_idf = term_freq * inverse_document_freq

  output = np.empty((len(words),3),dtype=object)
  output[:,0] = np.array(list(words.keys()),dtype=object)
  output[:,1] = tf_idf
  output[:,2] = np.array(list(words.values()),dtype=object)

  return output


if __name__=="__main__":

  # load data
  data = pd.read_csv('data/winemag-data-130k-v2.csv')

  # get reviews
  reviews = data['description']

  # get sentences
  # sentences = get_sentences(reviews)

  # save
  # with open('data/sentences.p', 'wb') as fp:
  #   pickle.dump(sentences, fp)

  try:
    # load words
    with open('data/words.p', 'rb') as fp:
      words = pickle.load(fp)

    # load bigrams
    with open('data/bigrams.p', 'rb') as fp:
      bigrams = pickle.load(fp)

  except:
    # bigrams
    wine2vec = Wine2Vec()
    words, bigrams = preprocess_reviews(reviews, wine2vec)

    # save words
    with open('data/words.p', 'wb') as fp:
      pickle.dump(words, fp)

    # save bigrams
    with open('data/bigrams.p', 'wb') as fp:
      pickle.dump(bigrams, fp)

  """ Compute TD-IDF """
  try:
    # load tfidf words
    with open('data/tfidf_words.p', 'rb') as fp:
      tfidf_words = pickle.load(fp)

    # load tfidf bigrams
    with open('data/tfidf_bigrams.p', 'rb') as fp:
      tfidf_bigrams = pickle.load(fp)

  except:
    # compute TF-IDF scores
    num_reviews = len(reviews)
    tfidf_words = compute_TF_IDF(words, num_reviews)
    tfidf_bigrams = compute_TF_IDF(bigrams, num_reviews)

    # save to pickle
    with open('data/tfidf_words.p', 'wb') as fp:
      pickle.dump(tfidf_words, fp)

    # save to pickle
    with open('data/tfidf_bigrams.p', 'wb') as fp:
      pickle.dump(tfidf_bigrams, fp)


  """ Convert to dict """
  flattened = [i for word in tfidf_words for i in word]
  words = flattened[::3]
  values = flattened[1::3]

  tf_idf_dict = dict()
  for idx, word in enumerate(words):
    tf_idf_dict[word] = values[idx]

  # save to pickle
  with open('output/tfidf_dict.p', 'wb') as fp:
    pickle.dump(tf_idf_dict, fp)  



