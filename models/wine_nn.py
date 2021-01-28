"""Train Neural Network"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import keras
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

ORIGINAL_DATA = "data/data.pkl"
TRAINING_DATA = "data/dfm.pkl"
OUTPUT_ROOT = "models/model_"

REMOVE_NANS = True


class WineNN:

    def __init__(self, targets):

        self.targets = targets
        self.encoder = LabelEncoder()
        self.input_size = None
        self.num_classes = None

        try:  # FIXME
            # load model
            self.model = keras.models.load_model(OUTPUT_ROOT + self.targets)
            self.input_size = self.model.layers[0].input_shape[1]

            # create image
            # keras.utils.plot_model(self.model, to_file='images/model.png', show_shapes=True)

            # load labels - TODO: Save encoder separately
            with open(ORIGINAL_DATA, 'rb') as fp:
                labels = (pickle.load(fp))[self.targets]

            if REMOVE_NANS:
                labels = labels.loc[labels.isnull() == 0]

            # convert to onehot
            self.encoder.fit(labels.values)

            # set trained
            self.trained = True

        except:
            self.trained = False
            print("No model was loaded.")
            self.load_data()
            print("The training data was loaded.")

    def load_data(self):
        # load features
        with open(TRAINING_DATA, 'rb') as fp:
            X = pickle.load(fp)

        # load labels
        with open(ORIGINAL_DATA, 'rb') as fp:
            labels = (pickle.load(fp))[self.targets]

        # convert to array and remove nan
        if REMOVE_NANS:
            nan_idx = labels.isnull()
            print("Removing " + str(sum(nan_idx)) +
                  " data points with NaN labels.")
            labels = labels.loc[nan_idx == 0]
            X = (X.values)[nan_idx == 0]

        # convert labels to onehot
        self.encoder.fit(labels.values)
        encoded_y = self.encoder.transform(labels.values)

        y = keras.utils.to_categorical(encoded_y)

        self.input_size = X.shape[1]
        self.num_classes = y.shape[1]

        # split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33)

    def create_model(self):

        self.trained = False

        # make model
        self.model = Sequential()
        self.model.add(
            Dense(512, input_dim=self.input_size, activation='sigmoid'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))
        self.model.add(Dense(128, activation='sigmoid'))
        self.model.add(Dropout(0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))
        self.model.add(BatchNormalization())
        self.model.add(Dense(512, activation='sigmoid'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        # compile
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, verbose=False):

        if self.trained:
            print("Model has already been trained.")
            return

        history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=64,
            epochs=25,
            validation_split=0.2,
            use_multiprocessing=True,
            verbose=verbose
        )

        self.trained = True

        # Save
        self.model.save(OUTPUT_ROOT + self.targets)

        result = self.model.evaluate(self.X_test, self.y_test, batch_size=64)
        if verbose:
            print("test loss, test acc:", result)

        return history, result

    def predict(self, features):
        if not self.trained:
            raise Exception("Model not trained.")

        input_features = features.reshape(1, self.input_size)
        prediction = self.model.predict(input_features)

        # return self.encoder.inverse_transform([np.argmax(prediction)])[0]
        return [self.encoder.classes_, prediction]


def plot_history(history):

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()


if __name__ == "__main__":

    targets = ['variety', 'province']
    histories = []

    for target in targets:
        model = WineNN(target)
        model.create_model()
        history, results = model.train(verbose=True)
        histories.append((history, results))

    with open('models/histories.pkl', 'wb') as fp:
        pickle.dump(histories, fp)
