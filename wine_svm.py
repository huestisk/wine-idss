"""Train SVM"""

import pandas as pd
import pickle

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABELS = "data/labels.p"
TRAINING_DATA = "data/dfm_words.p"
TARGETS = "variety"

class WineSVM:
  
  def __init__(self):

    try:
      # load model
      with open ("output/svm_" + TARGETS + ".p", 'rb') as fp:
        self.clf = pickle.load(fp) # TODO

      self.trained = True

    except:
      # make pipeline
      self.clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, verbose=1))
      self.trained = False

  def load_data(self, data_path, label_path):

    try:
      # load data
      with open (data_path, 'rb') as fp:
        tmp_data = pickle.load(fp)
        # check for NaN values
        if tmp_data.isnull().values.any():
          raise Exception("Features contain NaN values.")

      # load labels
      with open (label_path, 'rb') as fp:
        tmp_labels = pickle.load(fp)
        nan_idx = tmp_labels[TARGETS].isnull()

      # convet to array and remove nan
      self.data = (tmp_data.values)[nan_idx==0]
      self.labels = (tmp_labels[TARGETS].values)[nan_idx==0]

    except:
      raise Exception("Data could not be loaded.")

  def train(self):

    if self.trained:
      return self.clf

    print("Starting training...")
    self.clf.fit(self.data, self.labels)
    self.trained = True
    
    return self.clf

  def save(self, file_path):
    with open (file_path, 'wb') as fp:
      pickle.dump(self.clf, fp)


  def predict(self, features):
    if self.trained:
      return self.clf.predict(features)


if __name__=="__main__":
  
  svm = WineSVM()

  if not svm.trained:
    # load data
    svm.load_data(TRAINING_DATA, LABELS)

    # train svm
    svm.train()

    # save
    svm.save("output/svm_" + TARGETS + ".p")

  # use svm to predict
  features = []
  # prediction = svm.predict(features)
