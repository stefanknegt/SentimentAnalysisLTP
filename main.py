import numpy as np
import random
import re
import os
import io
import itertools
from os.path import isfile, join
#np.random.seed(10) #set seed before any keras import

from collections import Counter
from matplotlib import pyplot

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

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

def load_imdb_reviews():
    #Load the test and train data in seperate arrays
    train_positive_sentences = [l.strip() for l in open("train-pos.txt").readlines()]
    train_negative_sentences = [l.strip() for l in open("train-neg.txt").readlines()]
    train_positive_labels = [1 for sentence in train_positive_sentences]
    train_negative_labels = [0 for sentence in train_negative_sentences]

    test_positive_sentences = [l.strip() for l in open("test-pos.txt").readlines()]
    test_negative_sentences = [l.strip() for l in open("test-neg.txt").readlines()]
    test_positive_labels = [1 for sentence in test_positive_sentences]
    test_negative_labels = [0 for sentence in test_negative_sentences]

    """#Concatenate the sentences and labels for both train and test data
    test_sentences = np.concatenate([test_positive_sentences,test_negative_sentences], axis=0)
    test_labels = np.concatenate([test_positive_labels,test_negative_labels],axis=0)

    train_sentences = np.concatenate([train_positive_sentences,train_negative_sentences],axis=0)
    train_labels =  np.concatenate([train_positive_labels,train_negative_labels],axis=0)"""

    sentences = test_positive_sentences + test_negative_sentences + train_positive_sentences + train_negative_sentences
    labels = test_positive_labels + test_negative_labels + train_positive_labels + train_negative_labels

    #Randomly take 2000 reviews to match size of RT dataset
    rng_state = np.random.get_state()
    np.random.shuffle(sentences)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    sentences = sentences[:2000]
    labels = labels[:2000]

    assert(len(sentences) == len(labels))

    sentences = [clean_str(sent) for sent in sentences]
    sentences = [s.split(" ") for s in sentences]

    print("Loaded IMDB movie reviews")

    return sentences, labels

def load_imdb_reviews_full():
    #Load the test and train data in seperate arrays
    train_positive_sentences = [l.strip() for l in open("train-pos.txt").readlines()]
    train_negative_sentences = [l.strip() for l in open("train-neg.txt").readlines()]
    train_positive_labels = [1 for sentence in train_positive_sentences]
    train_negative_labels = [0 for sentence in train_negative_sentences]

    test_positive_sentences = [l.strip() for l in open("test-pos.txt").readlines()]
    test_negative_sentences = [l.strip() for l in open("test-neg.txt").readlines()]
    test_positive_labels = [1 for sentence in test_positive_sentences]
    test_negative_labels = [0 for sentence in test_negative_sentences]

    """#Concatenate the sentences and labels for both train and test data
    test_sentences = np.concatenate([test_positive_sentences,test_negative_sentences], axis=0)
    test_labels = np.concatenate([test_positive_labels,test_negative_labels],axis=0)

    train_sentences = np.concatenate([train_positive_sentences,train_negative_sentences],axis=0)
    train_labels =  np.concatenate([train_positive_labels,train_negative_labels],axis=0)"""

    sentences = test_positive_sentences + test_negative_sentences + train_positive_sentences + train_negative_sentences
    labels = test_positive_labels + test_negative_labels + train_positive_labels + train_negative_labels

    """#Randomly take 2000 reviews to match size of RT dataset
    rng_state = np.random.get_state()
    np.random.shuffle(sentences)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    sentences = sentences[:2000]
    labels = labels[:2000]"""

    assert(len(sentences) == len(labels))

    sentences = [clean_str(sent) for sent in sentences]
    sentences = [s.split(" ") for s in sentences]

    print("Loaded IMDB movie reviews full")

    return sentences, labels

def load_rottentomatoes_reviews():
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

    print("Loaded Rotten Tomatoes movie reviews")
    return [sentences, labels]

def pad_sentences(sentences, padding_word="</s>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """

    longest_review= max(len(x) for x in sentences)
    print("Max review length is: %d " % longest_review)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = longest_review - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)

    return padded_sentences

    """sequence_length = 500 #max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if(len(sentence) <= sequence_length):
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        else:
            new_sentence = sentence[0:sequence_length]
            padded_sentences.append(new_sentence)
    return padded_sentences"""

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
    #model.add(Dropout(0.5)) #location of dropout does not really influence acc (before or after flatten)
    #Flatten the 32x500 to one dimension
    model.add(Flatten())
    model.add(Dropout(0.2))
    #Add hidden layer with 250 nodes
    model.add(Dense(250, activation='relu'))
    #Add output layer with 1 node to output either 0 or 1
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Fit the model (only 2 epochs are used since overfitting is a problem)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=2) # 128 > 64 > 32 for RT
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

def Conv_embedding(x_train, y_train, x_test, y_test,vocab_size,sentence_size):
    # create the model
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=sentence_size))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')) #this layers increases performance with load_data2 for RT
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=32, verbose=1) #batch size 32 works better than 64 which works slightly better than 128 for RT
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

def LSTM_embedding(x_train, y_train, x_test, y_test,vocab_size,sentence_size):
    # Create the model
    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=sentence_size))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)

    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

sentences, labels = load_imdb_reviews_full() #geeft nu 89.58 en ??
#sentences, labels = load_imdb_reviews() #geeft nu 65 en 78%
#sentences, labels = load_rottentomatoes_reviews() #geeft nu 74,5 en 88%

padded_sentence = pad_sentences(sentences)
vocabulary, vocabulary_inv = build_vocab(padded_sentence)
x, y = build_input_data(padded_sentence, labels, vocabulary)

vocab_size = len(vocabulary)

# randomly shuffle data
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

#x_shuffled = x_shuffled[:2000]
#y_shuffled = y_shuffled[:2000]

# split train/dev set
cutoff = 0.1*len(x_shuffled) #10% test and 90% train
print("Cutoff is %d" % cutoff)
x_train, x_test = x_shuffled[:-int(cutoff)], x_shuffled[-int(cutoff):]
y_train, y_test = y_shuffled[:-int(cutoff)], y_shuffled[-int(cutoff):]

print("The distribution of pos and neg in train data is %.2f %.2f" % (float(np.count_nonzero(y_train)/len(y_train)),(1-float(np.count_nonzero(y_train)/len(y_train)))))
print("The number of positive reviews in the train data is %d " % np.count_nonzero(y_train))

print(x_train.shape)
sentence_size = x_train.shape[1]

print("Done loading data..")

print ('Train/Dev split: %d/%d' % (len(y_train), len(y_test)))
print ('train shape:', x_train.shape)
print ('dev shape:', x_test.shape)
print ('vocab_size', vocab_size)
print ('sentence max words', sentence_size)

MLP_embedding(x_train,y_train,x_test,y_test,vocab_size,sentence_size)
Conv_embedding(x_train,y_train,x_test,y_test,vocab_size,sentence_size)
#LSTM_embedding(x_train,y_train,x_test,y_test,vocab_size,sentence_size)
