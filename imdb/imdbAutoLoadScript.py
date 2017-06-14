__author__ = "Stefan Knegt"

"""
A first version of sentiment analysis for the course language technology project 16/17
"""

import numpy as np
np.random.seed(100) #set seed before any keras import

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.datasets import imdb

from matplotlib import pyplot

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

def load_data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data()
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    return x,y,x_train, y_train, x_test, y_test

def data_stats(x,y):
    print("Data: ")
    print(x.shape)
    print(y.shape)
    print(x[1])

    print("Classes: ")
    print(np.unique(y))

def movie_review_information(X):
    # Summarize review length
    print("Review length: ")
    result = [len(x) for x in X]
    print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))
    # plot review length
    pyplot.boxplot(result)
    pyplot.show()

def load_data_word_embeddings(number_words,length_review):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=number_words)

    x_train = sequence.pad_sequences(x_train, maxlen=length_review)
    x_test = sequence.pad_sequences(x_test, maxlen=length_review)

    print(x_train.shape)
    print(x_test.shape)

    return x_train, y_train, x_test, y_test

def MLP_embedding(x_train, y_train, x_test, y_test):
    # create the model
    model = Sequential()
    #Embedding layer gives as output a 32x500 matrix
    model.add(Embedding(5000, 32, input_length=500)) #vocab size 5000, word vector size 32,input length 500
    #Flatten the 32x500 to one dimension
    model.add(Flatten())
    #Add hidden layer with 250 nodes
    model.add(Dense(250, activation='relu'))
    #Add output layer with 1 node to output either 0 or 1
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Fit the model (only 2 epochs are used since overfitting is a problem)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

def Conv_embedding(x_train, y_train, x_test, y_test):
    # create the model
    model = Sequential()
    model.add(Embedding(5000, 32, input_length=500))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

def LSTM_imdb(x_train, y_train, x_test, y_test):
    # Create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(5000, embedding_vector_length, input_length=500))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)

    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

x,y,_,_,_,_ = load_data()
data_stats(x,y)
movie_review_information(x)
x_train, y_train, x_test, y_test = load_data_word_embeddings(5000,500)

#MLP_embedding(x_train, y_train, x_test, y_test) #Gives +- 87% accuracy
Conv_embedding(x_train, y_train, x_test, y_test) #Gives +- 88% accuracy
#LSTM_imdb(x_train, y_train, x_test, y_test) #Gives +-
