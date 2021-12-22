import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.python.keras.backend import sparse_categorical_crossentropy
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.core import Dropout

DATASET_PATH = "data.json"


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert list into numpy array
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets


def plot_history(history):
    fig, axs = plt.subplots(2)

    # accuracy subplot
    axs[0].plot(history.history["accuracy"], label="MLP Train Accuracy")
    axs[0].plot(history.history["val_accuracy"], label="MLP Test Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Evaluation")

    axs[1].plot(history.history["loss"], label="MLP Train Error")
    axs[1].plot(history.history["val_loss"], label="MLP Test Error")
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
    # load data
    inputs, targets = load_data(DATASET_PATH)

    #print (inputs.shape)

    # split the data into train and test data
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,
                                                                              targets,
                                                                              test_size=0.3)

    # build the network architecture
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # first hidden layer
        keras.layers.Dense(512, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # second hidden layer
        keras.layers.Dense(256, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        # third hidden layer
        keras.layers.Dense(64, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss=sparse_categorical_crossentropy, metrics=["accuracy"])
    model.summary()

    # train the network
    history = model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), epochs=100,
                        batch_size=32)
    test_error, test_accuracy = model.evaluate(
        inputs_test, targets_test, verbose=1)
    print("Accuracy on test set is {}".format(test_accuracy))

    # Make prediction on a sample
    X = inputs_test[50]
    y = targets_test[50]
    predict(model, X, y)

    plot_history(history)
