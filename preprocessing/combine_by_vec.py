
import multiprocessing
import gensim.models.word2vec as w2v

import pickle

from collections import Counter

NUM_FEATURES = 300
MIN_WORD_COUNT = 10
NUM_WORKERS = multiprocessing.cpu_count()
CONTEXT_SIZE = 10
DOWNSAMPLING = 1e-3
SEED=1993

COMBINATION_THRESHOLD=0.5
COUNT_THRESHOLD=0.7

class Wine2Vec:

  def __init__(self):

    self.trained = True
    self.threshold = COMBINATION_THRESHOLD
    self.count_threshold = COUNT_THRESHOLD

    try:
      self.wine2vec = w2v.Word2Vec.load("output/wine2vec.model")
    except:
      self.trained = False
      self.wine2vec = w2v.Word2Vec(
        sg=1,
        seed=SEED,
        workers=NUM_WORKERS,
        size=NUM_FEATURES,
        min_count=MIN_WORD_COUNT,
        window=CONTEXT_SIZE,
        sample=DOWNSAMPLING
      )

  def build_vocab(self, sentences):
    self.wine2vec.build_vocab(sentences)
    # print('Word2Vec vocabulary length:', len(self.wine2vec.wv.vocab))
    # print(self.wine2vec.corpus_count)

  def train(self, sentences):
    self.wine2vec.train(sentences, total_examples=self.wine2vec.corpus_count, epochs=self.wine2vec.epochs)
    self.trained = True

  def convert_combinable(self, combinable_words):

    # Create dictionary
    conversion_map = dict()
    for _, values in combinable_words.items():

      flattened = [i for value in values for i in value]
      names = flattened[::3]
      values = flattened[1::3]
      counts = flattened[2::3]

      max_idx = counts.index(max(counts))

      for idx, name in enumerate(names):
        if name in conversion_map:
          conversion_map[name].append((names[max_idx], counts[max_idx], values[idx]))
        else:
          conversion_map[name] = [(names[max_idx], counts[max_idx], values[idx])]

    # Collapse dictionary
    for key in conversion_map.keys():

      flattened = [i for value in conversion_map[key] for i in value]
      names = flattened[::3]
      values = flattened[1::3]
      counts = flattened[2::3]

      max_count = counts.index(max(counts))
      max_value = values.index(max(values))

      if max_count == max_value:
        conversion_map[key] = names[max_count]

      elif counts[max_count] > self.count_threshold*sum(counts):
        conversion_map[key] = names[max_count]

      else:
        conversion_map[key] = names[max_value]

    return conversion_map

  def get_combinable(self, sentences):
    """for each unique words combine when possible"""
    if not self.trained:
      raise Exception("The model has not been trained.")
    else:
      # Flatten
      words = [word for sentence in sentences for word in sentence]
      # Count
      counted_words = Counter(words)
      # Combinable
      all_combinable_words = dict()
      for word in words:
        # skip, if already done
        if word in all_combinable_words:
          continue
        # get most similar words
        try:
          similar_words = self.most_similar(word)
        except KeyError:
          # all_combinable_words[word] = []
          continue
        # get combinable words
        combinable_words = [[word, 1.0, counted_words[word]]]
        for similar_word, value in similar_words:
          if similar_word in words and value > self.threshold:
            count = counted_words[similar_word]
            combinable_words.append([similar_word, value, count])
        # add to dictionary
        all_combinable_words[word] = combinable_words

      return all_combinable_words

  def most_similar(self, word):
    return self.wine2vec.wv.most_similar(word)

  def save_model(self):
    self.wine2vec.save("output/wine2vec.model")



if __name__ == "__main__":

  # load sentences
  with open ('data/sentences.p', 'rb') as fp:
    sentences = pickle.load(fp)

  # create class
  wine2vec = Wine2Vec()

  if wine2vec.trained:
    # test
    # print(wine2vec.most_similar('aroma'))

    try:
      # load combinable words
      with open ('data/combinable_words.p', 'rb') as fp:
        combinable_words = pickle.load(fp)

      # create conversion map
      conversion_map = wine2vec.convert_combinable(combinable_words)

      # save
      with open('data/conversion_map.p', 'wb') as fp:
        pickle.dump(conversion_map, fp)

    except: # if file doesnt exist
      # combine
      combinable_words = wine2vec.get_combinable(sentences)

      # save
      with open('data/combinable_words.p', 'wb') as fp:
        pickle.dump(combinable_words, fp)
 
  else:
    # build vocab
    wine2vec.build_vocab(sentences)
  
    # train
    wine2vec.train(sentences)
  
    # test
    print(wine2vec.most_similar('aroma'))
  
    # save
    wine2vec.save_model()