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
from nltk.corpus import stopwords
import time

class TopicSemanticAnalysis:

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def topicWeightingAddition(self, topic, weight):
        listOfTopics = []

        for i in listOfTopics:
            continue

    
    def featureCreator(self, features, labels):
        all_words = []
        documents = []


        stop_words = list(set(stopwords.words('english')))

        #  using keys of nltk: j = adjective, v = verb and r is adverb
        only_words_allowed = ["J"] # we are only concerned with adjectives


        for x in range(len(self.features)):
            documents.append((self.features[x], self.labels[x]))

            cleaned = re.sub(r'[^(a-zA-Z)\s]','', str(self.features[x]))
            
            tokenized = word_tokenize(cleaned)
            stopped = [w for w in tokenized if not w in stop_words]
            pos = nltk.pos_tag(stopped)
            
        #make a list of  all adjectives identified by the allowed word types list above
            for w in pos:
                if w[1][0] in only_words_allowed:
                    all_words.append(w[0].lower())
        
        return all_words, documents

    def get_word_features(self, allwordslist):
        BOW = nltk.FreqDist(allwordslist)
        word_features = list(BOW.keys())[:5000]
        word_features[0], word_features[-1]

        save_word_features = open("./Datasets/saved/word_features.pickle","wb")
        pickle.dump(word_features, save_word_features)

        return word_features

    def find_features(self, document, word_features):
        words = word_tokenize(str(document))
        features = {}
        for w in word_features:
            features[w] = (w in words)

        return features

    def train_val_generator(self, docs, wordfeats):

        # Features mapped to each tweet
        featuresets = [(self.find_features(rev, wordfeats), category) for (rev, category) in docs]

        # Shuffling the documents 
        random.shuffle(featuresets)
        train_set = 0.8 * len(featuresets)
        training_set = featuresets[:int(train_set)]
        validation_set = featuresets[int(train_set):]

        return training_set, validation_set
    """
    The following below are all the classes that are implemented as part of the machine learning models

    NBC : Naive Bayes 
    MNB : Multinomial Naive Bayes
    BNB : Bernoulli Naive Bayes *** Highest performing one out of all tests
    Log : Logistic regression classifier
    SGD : Stochastic Gradient classifier

    """

    """

    Each function implementation of the following models also saves the best performing model to the Datasets/saved_models directory. 
    Time is also recorded per computation for better visibility around the balance between accuracy + computation

    """
    def NBClf(self, training_set, validation_set):
        start = time.time()
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, validation_set))*100)
        classifier.show_most_informative_features(15)
        end = time.time() - start
        print('Naive Bayes took: ' + str(end) + ' seconds')
        self.create_pickle(classifier, '../Datasets/saved_models/ONB_clf.pickle' )
        acc_score = nltk.classify.accuracy(classifier, validation_set)*100
        return acc_score

    def MNBClf(self, training_set, validation_set):
        start = time.time()
        MNB_clf = SklearnClassifier(MultinomialNB())
        MNB_clf.train(training_set)
        print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_clf, validation_set))*100)
        self.create_pickle(MNB_clf, '../Datasets/saved_models/MNB_clf.pickle' )
        end = time.time() - start
        print('Multinomial Naive Bayes took: ' + str(end) + ' seconds')
        acc_score = nltk.classify.accuracy(MNB_clf, validation_set)*100
        return acc_score

    def BNBClf(self, training_set, validation_set):
        start = time.time()
        BNB_clf = SklearnClassifier(BernoulliNB())
        BNB_clf.train(training_set)
        print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_clf, validation_set))*100)
        end = time.time() - start
        print(str(end) + ' seconds')
        self.create_pickle(BNB_clf, '../Datasets/saved_models/BNB_clf.pickle' )
        acc_score = nltk.classify.accuracy(BNB_clf, validation_set)*100
        return acc_score

    def LogClf(self, training_set, validation_set):
        start = time.time()
        LogReg_clf = SklearnClassifier(LogisticRegression())
        LogReg_clf.train(training_set)
        print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogReg_clf, validation_set))*100)
        end = time.time() - start
        print(str(end) + ' seconds')
        self.create_pickle(LogReg_clf, '../Datasets/saved_models/LogReg_clf.pickle' )
        acc_score = nltk.classify.accuracy(LogReg_clf, validation_set)*100
        return acc_score

    def SGDClf(self, training_set, validation_set):
        start = time.time()
        SGD_clf = SklearnClassifier(SGDClassifier())
        SGD_clf.train(training_set)
        print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGD_clf, validation_set))*100)
        end = time.time() - start
        print(str(end) + ' seconds')
        self.create_pickle(SGD_clf, '../Datasets/saved_models/SGD_clf.pickle' )
        acc_score = nltk.classify.accuracy(SGD_clf, validation_set)*100
        return acc_score

    def create_pickle(self, c, file_name): 
        save_classifier = open('Datasets/'+file_name, 'wb')
        pickle.dump(c, save_classifier)
        save_classifier.close()