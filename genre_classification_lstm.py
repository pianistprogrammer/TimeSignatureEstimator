import os
import json
from re import X
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import sparse_categorical_crossentropy

DATASET_PATH = "data.json"


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert list into numpy array
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y


def prepare_dataset(test_size, valisation_size):

    #load the data
    X, y = load_data(DATASET_PATH)

    #create the train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)

    #create the train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=valisation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):

    #create the model
    model = keras.Sequential()

    #first lstm layer
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))

    #second lstm layer
    model.add(keras.layers.LSTM(64))

    #Dense layer
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    #output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


def plot_history(history):
    fig, axs = plt.subplots(2)

    # accuracy subplot
    axs[0].plot(history.history["accuracy"], label="LSTM Train Accuracy")
    axs[0].plot(history.history["val_accuracy"], label="LSTM Test Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Evaluation")

    axs[1].plot(history.history["loss"], label="LSTM Train Error")
    axs[1].plot(history.history["val_loss"], label="LSTM Test Error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error Evaluation")

    plt.show()


def predict(model, X, y):

    X = X[np.newaxis, ...]

    prediction = model.predict(X)
    print("Prediction shape is: {}".format(prediction.shape))

    predicted_index = np.argmax(prediction, axis=1)
    print("Expected Index is {}, Predicted Index is {}".format(y, predicted_index))


if __name__ == "__main__":
    #Create train, validation and test set
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset(
        0.25, 0.2)

    #Build the CNN network
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    #Compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss=sparse_categorical_crossentropy, metrics=["accuracy"])
    model.summary()

    #Train  the network
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=40,
              batch_size=32)
    #Evaluate the lstm on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is {}".format(test_accuracy))

    #Make prediction on a sample
    X = X_test[50]
    y = y_test[50]
    predict(model, X, y)

    plot_history(history)
