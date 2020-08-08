
# Import the required packages
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from matplotlib import pyplot
import time as t
import psutil

def mlp():
    '''
    This function is the implemenatation of
    MultiLayer Perceptron achriture.
    :return:
    '''
    dataset = pd.read_csv("ES_dataset_rand.csv", delimiter=",")
    X_train = dataset.iloc[0:280, 0:7].values
    X_test = dataset.iloc[280:, 0:7].values
    Y_train = dataset.iloc[0:280, 7:8].values
    Y_test = dataset.iloc[280:, 7:8].values
    model = Sequential()
    model.add(Dense(32, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, validation_split=0.33, epochs=35)
    print(model.evaluate(X_test, Y_test))
    predictions_test = model.predict(X_test)
    print(predictions_test[0])
    model.summary()
    plot_training_acc(history)
    plot_training_loss(history)
    plot_testing_acc(history)
    plot_testing_loss(history)


# Graph Plotting

def plot_training_acc(history):
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Epochs')
    pyplot.plot(history.history['acc'])
    pyplot.title('Training Data : Accuracy Vs Epochs')
    pyplot.show()


def plot_training_loss(history):
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epochs')
    pyplot.plot(history.history['loss'])
    pyplot.title('Training Data : Loss Vs Epochs')
    pyplot.show()


def plot_testing_acc(history):
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Epochs')
    pyplot.plot(history.history['val_acc'])
    pyplot.title('Testing Data : Accuracy Vs Epochs')
    pyplot.show()


def plot_testing_loss(history):
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epochs')
    pyplot.plot(history.history['val_loss'])
    pyplot.title('Testing Data : Loss Vs Epochs')
    pyplot.show()


if __name__ == '__main__':
    t1 = t.time()
    mlp()
    t2 = t.time()
    print("Time taken : ", t2 - t1)
    print("CPU Consumption : ", psutil.cpu_percent())
