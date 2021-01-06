
import multiprocessing
import gensim.models.word2vec as w2v

import pickle

NUM_FEATURES = 300
MIN_WORD_COUNT = 10
NUM_WORKERS = multiprocessing.cpu_count()
CONTEXT_SIZE = 10
DOWNSAMPLING = 1e-3
SEED=1993

class Wine2Vec:

  def __init__(self):
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

  def most_similar(self, word):
    return self.wine2vec.wv.most_similar(word)

  def save(self):
    self.wine2vec.save("output/wine2vec.model")

if __name__ == "__main__":

  # load sentences
  with open ('data/sentences.p', 'rb') as fp:
    sentences = pickle.load(fp)

  # create class
  wine2vec = Wine2Vec()

  # build vocab
  wine2vec.build_vocab(sentences)

  # train
  wine2vec.train(sentences)

  # test
  print(wine2vec.most_similar('berri'))

  # save
  wine2vec.save()