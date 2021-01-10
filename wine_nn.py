"""Train Neural Network"""

import pandas as pd
import numpy as np
import pickle

import keras
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

LABELS = "data/labels.p"
TRAINING_DATA = "data/dfm_words.p"
TARGETS = "province"

class WineNN:
  
  def __init__(self, targets):

    self.targets = targets
    self.encoder = LabelEncoder()
    self.input_size = None
    self.num_classes = None

    # training data
    self.X_train = self.X_test = self.y_train = self.y_test = []

    try:
      # load model
      self.model = keras.models.load_model("output/model_" + self.targets)
      self.input_size = self.model.layers[0].input_shape[1]

      # load labels
      with open ("data/labels.p", 'rb') as fp:
        tmp_labels = pickle.load(fp)
        nan_idx = tmp_labels[self.targets].isnull()

      # convert to array and remove nan
      tmp_labels = tmp_labels.loc[nan_idx==0][self.targets]

      # convert to onehot
      self.encoder.fit(tmp_labels.values)

      # set trained
      self.trained = True

    except:
      self.trained = False


  def create_model(self):
    
    self.trained = False

    # make model - SVM
    self.model = Sequential()
    self.model.add(Dense(300, input_dim=self.input_size, activation='relu'))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(300, activation='relu'))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(self.num_classes, activation='softmax'))

    # compile
    self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
      

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
        nan_idx = tmp_labels[self.targets].isnull()

      # convert to array and remove nan
      tmp_labels = tmp_labels.loc[nan_idx==0][self.targets]
      X = (tmp_data.values)[nan_idx==0]

      # convert to onehot
      self.encoder.fit(tmp_labels.values)
      encoded_y = self.encoder.transform(tmp_labels.values)

      y = keras.utils.to_categorical(encoded_y)

      self.input_size = X.shape[1]
      self.num_classes = y.shape[1]

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

    results = self.model.evaluate(self.X_test, self.y_test, batch_size=64)
    print("test loss, test acc:", results)

  def save(self, file_path):
    self.model.save(file_path)

  def predict(self, features):
    if not self.trained:
      raise Exception("Model not trained.")

    input_features = features.reshape(1, self.input_size)
    prediction = self.model.predict(input_features)

    return [self.encoder.classes_, prediction]# self.encoder.inverse_transform([np.argmax(prediction)])[0]


if __name__=="__main__":
  
  model = WineNN(TARGETS)

  if not model.trained:

    # load data
    model.load_data(TRAINING_DATA, LABELS)

    # train model
    model.create_model()
    model.train(verbose=True)

    # save
    model.save("output/model_" + TARGETS)

  # # use model to predict
  # features = []
  # prediction = model.predict(features)
