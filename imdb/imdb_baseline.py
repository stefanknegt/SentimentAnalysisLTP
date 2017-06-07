from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

import numpy as np
import random

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

    #Concatenate the sentences and labels for both train and test data
    test_sentences = np.concatenate([test_positive_sentences,test_negative_sentences], axis=0)
    test_labels = np.concatenate([test_positive_labels,test_negative_labels],axis=0)

    train_sentences = np.concatenate([train_positive_sentences,train_negative_sentences],axis=0)
    train_labels =  np.concatenate([train_positive_labels,train_negative_labels],axis=0)

    #Shuffle the train and test data and return the x_train,y_train,x_test,y_test lists
    random.seed(113)

    assert(len(train_sentences)==len(train_labels))
    train_data = list(zip(train_sentences,train_labels))
    random.shuffle(train_data)

    assert(len(test_sentences)==len(test_labels))
    test_data = list(zip(test_sentences,test_labels))
    random.shuffle(test_data)

    x_train = [sentence for sentence, label in train_data]
    y_train = [label for sentence, label in train_data]

    x_test = [sentence for sentence, label in test_data]
    y_test = [label for sentence, label in test_data]

    return x_train,y_train,x_test,y_test


x_train,y_train,x_test,y_test = load_imdb_reviews()

#Make vectorizer
vectorizer = CountVectorizer()

# CLASSIFIER 1: Naive Bayes classifier for multinomial models
classifier = Pipeline( [('vec', vectorizer),
                        ('clf', MultinomialNB()),])
classifier = classifier.fit(x_train, y_train)
baseline_test = classifier.predict(x_test)
#print(np.mean(baseline_test == y_test))

#CLASSIFIER 2: Logistic Regression
classifier = Pipeline( [('vec', vectorizer),
                        ('clf', LogisticRegression())] )
classifier = classifier.fit(x_train, y_train)
y_predicted_test = classifier.predict(x_test)
#print(np.mean(y_predicted_test == y_test))

#Print the accuracy of both classifiers
accuracy_test = accuracy_score(y_test, y_predicted_test)

print("===== test set ====")
print("Baseline:   {0:.2f}".format(accuracy_score(y_test, baseline_test)*100))
print("Classifier: {0:.2f}".format(accuracy_test*100))
