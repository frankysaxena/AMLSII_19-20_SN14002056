import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import re

all_words = []
documents = []

from nltk.corpus import stopwords
import re

stop_words = list(set(stopwords.words('english')))

#  using keys of nltk: j = adjective, v = verb and r is adverb
allowed_word_types = ["J"] # we are only concerned with adjectives


for p in range(len(x_train)):
    documents.append((x_train[p], y_train[p]))

    cleaned = re.sub(r'[^(a-zA-Z)\s]','', str(x_train[p]))
    
    tokenized = word_tokenize(cleaned)
    stopped = [w for w in tokenized if not w in stop_words]
    pos = nltk.pos_tag(stopped)
    
#     make a list of  all adjectives identified by the allowed word types list above
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

def find_features(document):
    words = word_tokenize(str(document))
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# Features mapped to each tweet
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# Shuffling the documents 
random.shuffle(featuresets)