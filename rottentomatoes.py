__author__ = "Stefan Knegt"

"""
A first version of sentiment analysis for the course language technology project 16/17
"""


import sys
import os
import time
import io
import random
import re
from os.path import isfile, join


import numpy as np
#np.random.seed(113) #set seed before any keras import

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D, AveragePooling1D

from keras.datasets import imdb

from matplotlib import pyplot

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

import itertools
from collections import Counter

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    data_dir = "rt-polaritydata/rt-polaritydata/rt-polarity."
    classes = ['pos', 'neg']
    # Read the data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    labels = []
    for curr_class in classes:
        filename = data_dir + curr_class + ".txt"
        with io.open(filename, encoding='latin-1') as f:
            content = f.readlines()
            if(curr_class == 'pos'):
                pos_sentences = [x.strip() for x in content]
            else:
                neg_sentences = [x.strip() for x in content]
            if curr_class == 'pos':
                labels.extend([1 for i in range(0,len(pos_sentences))])
            else:
                labels.extend([0 for i in range(0,len(neg_sentences))])

    sentences = pos_sentences + neg_sentences
    sentences = [clean_str(sent) for sent in sentences]
    sentences = [s.split(" ") for s in sentences]

    return [sentences, labels]

def load_data_and_labels2():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    pos_path = "review_polarity/txt_sentoken/pos"
    neg_path = "review_polarity/txt_sentoken/neg"
    pos_data_dir = os.listdir(pos_path)
    neg_data_dir = os.listdir(neg_path)
    pos_sentences = []
    neg_sentences = []
    labels = []

    pos_files = [os.path.join(pos_path,f) for f in pos_data_dir if isfile(join(pos_path, f))]

    for i in pos_files:
        with io.open(i, encoding='latin-1') as f:
            content = f.read()
            pos_sentences.append(content)

    #print(pos_sentences[0])

    neg_files = [os.path.join(neg_path,f) for f in neg_data_dir if isfile(join(neg_path, f))]

    for i in neg_files:
        with io.open(i, encoding='latin-1') as f:
            content = f.read()
            neg_sentences.append(content)

    #print(neg_sentences[0])

    for i in range(0,2000):
        if(i < 1000):
            labels.append(1)
        if(i > 999 and i < 2000):
            labels.append(0)

    sentences = pos_sentences + neg_sentences
    #print(all(isinstance(i, str) for i in sentences))
    sentences = [clean_str(sent) for sent in sentences]
    sentences = [s.split(" ") for s in sentences]

    assert(len(labels) == len(sentences))
    return [sentences, labels]


def pad_sentences(sentences, padding_word="</s>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def MLP_embedding(x_train, y_train, x_test, y_test, vocab_size, sentence_size):
    # create the model
    model = Sequential()
    #Embedding layer gives as output a 32x500 matrix
    model.add(Embedding(vocab_size, 32, input_length=sentence_size)) #word vector size 32
    model.add(Dropout(0.5))
    #Flatten the 32x500 to one dimension
    model.add(Flatten())
    #Add hidden layer with 250 nodes
    model.add(Dense(250, activation='relu'))
    #Add output layer with 1 node to output either 0 or 1
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Fit the model (only 2 epochs are used since overfitting is a problem)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=128, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

def Conv_embedding(x_train, y_train, x_test, y_test,vocab_size,sentence_size):
    # create the model
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=sentence_size))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

def LSTM_embedding(x_train, y_train, x_test, y_test,vocab_size,sentence_size):
    # Create the model
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=sentence_size))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)

    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


# Load and preprocess data
#sentences, labels = load_data_and_labels()
sentences, labels = load_data_and_labels2()
sentences_padded = pad_sentences(sentences)
vocabulary, vocabulary_inv = build_vocab(sentences_padded)
x, y = build_input_data(sentences_padded, labels, vocabulary)
print(x.shape)
print(x[1])

vocab_size = len(vocabulary)

# randomly shuffle data
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# split train/dev set
# there are a total of 10662 labeled examples to train on

#x_train, x_test = x_shuffled[:-1000], x_shuffled[-1000:]
#y_train, y_test = y_shuffled[:-1000], y_shuffled[-1000:]

x_train, x_test = x_shuffled[:-200], x_shuffled[-200:]
y_train, y_test = y_shuffled[:-200], y_shuffled[-200:]


sentence_size = x_train.shape[1]

print ('Train/Dev split: %d/%d' % (len(y_train), len(y_test)))
print ('train shape:', x_train.shape)
print ('dev shape:', x_test.shape)
print ('vocab_size', vocab_size)
print ('sentence max words', sentence_size)

#Conv_embedding(x_train,y_train,x_test,y_test,vocab_size,sentence_size) Max attained so far is 87% for load_data2
LSTM_embedding(x_train,y_train,x_test,y_test,vocab_size,sentence_size)
