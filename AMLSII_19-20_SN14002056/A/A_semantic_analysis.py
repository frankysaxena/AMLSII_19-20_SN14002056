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

class SemanticAnalysis:

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels


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
            
        #     make a list of  all adjectives identified by the allowed word types list above
            for w in pos:
                if w[1][0] in only_words_allowed:
                    all_words.append(w[0].lower())
        
        return all_words

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
train_set = 0.8 * len(featuresets)
training_set = featuresets[:int(train_set)]

testing_set = featuresets[int(train_set):]
print( 'training_set :', len(training_set), '\ntesting_set :', len(testing_set))

start = time.time()

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)
end = time.time() - start
print(str(end) + ' seconds')

start = time.time()


MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_clf, testing_set))*100)

end = time.time() - start
print(str(end) + ' seconds')


start = time.time()

BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_clf, testing_set))*100)

end = time.time() - start
print(str(end) + ' seconds')


start = time.time()

LogReg_clf = SklearnClassifier(LogisticRegression())
LogReg_clf.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogReg_clf, testing_set))*100)

end = time.time() - start
print(str(end) + ' seconds')


start = time.time()

SGD_clf = SklearnClassifier(SGDClassifier())
SGD_clf.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGD_clf, testing_set))*100)

end = time.time() - start
print(str(end) + ' seconds')


def create_pickle(c, file_name): 
    save_classifier = open('Datasets/'+file_name, 'wb')
    pickle.dump(c, save_classifier)
    save_classifier.close()

classifiers_dict = {'ONB': [classifier, 'saved_models/ONB_clf.pickle'],
                    'MNB': [MNB_clf, 'saved_models/MNB_clf.pickle'],
                    'BNB': [BNB_clf, 'saved_models/BNB_clf.pickle'],
                    'LogReg': [LogReg_clf, 'saved_models/LogReg_clf.pickle'],
                    'SGD': [SGD_clf, 'saved_models/SGD_clf.pickle']}

for clf, listy in classifiers_dict.items(): 
    create_pickle(listy[0], listy[1])