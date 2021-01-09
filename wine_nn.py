"""Train Neural Network"""

import pandas as pd
import pickle

import keras
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

from sklearn.model_selection import train_test_split

LABELS = "data/labels.p"
TRAINING_DATA = "data/dfm_words.p"
TARGETS = "variety"
INPUT_SIZE = 500

class WineNN:
  
  def __init__(self):

    try:
      # load model
      self.model = keras.models.load_model("output/model_" + TARGETS)
      self.trained = True

    except:
      # make model - SVM
      self.trained = False
      self.model = Sequential()
      self.model.add(Dense(INPUT_SIZE, activation='relu'))
      self.model.add(Dense(1, kernel_regularizer=l2(0.01)))
      self.model.add(Activation('softmax'))

      # compile
      self.model.compile(loss='squared_hinge',
                        optimizer='adadelta',
                        metrics=['accuracy'])
      # training data
      self.X_train = self.X_test = self.y_train = self.y_test = []
      

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

      # convert to array and remove nan
      tmp_labels = tmp_labels.loc[nan_idx==0][TARGETS]
      onehot = pd.get_dummies(tmp_labels)
      self.label_names = onehot.columns

      X = (tmp_data.values)[nan_idx==0]
      y = onehot.values

      # split data
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33)

    except:
      raise Exception("Data could not be loaded.")

  def train(self, verbose=False):

    if self.trained:
      return

    self.model.fit(
        self.X_train, 
        self.y_train,
        batch_size=64,
        epochs=20,
        verbose=verbose
      )

    self.trained = True

    results = self.model.evaluate(self.X_test, self.y_test, batch_size=128)
    print("test loss, test acc:", results)


  def save(self, file_path):
    self.model.save(file_path)


  def predict(self, features):
    if self.trained:
      return self.model.predict(features)


if __name__=="__main__":
  
  model = WineNN()

  if not model.trained:
    # load data
    model.load_data(TRAINING_DATA, LABELS)

    # train model
    model.train(verbose=True)

    # save
    model.save("output/model_" + TARGETS)

  # use model to predict
  features = []
  # prediction = model.predict(features)
