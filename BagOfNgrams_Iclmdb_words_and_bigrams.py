
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LogReg
#from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from time import time
from sklearn.metrics import classification_report
import gensim
from collections import namedtuple
SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

def Classification(classifier, train, train_labels, test, test_labels):
    grid_search = GridSearchCV(classifiers_dict[classifier], param_grid = search_parameters[classifier], error_score=0.0, n_jobs = -1)
    t0 = time()
    grid_search.fit(train, train_labels)
    print("done in %0.3fs" % (time() - t0))
    #print("Best score: %0.3f" % grid_search.best_score_)
    best_parameters = grid_search.best_estimator_.get_params()
    k = ""
    for param_name in sorted(search_parameters[classifier].keys()):
        #print("%s: %r" % (param_name, best_parameters[param_name]))
        k += "%s: %r\n" % (param_name, best_parameters[param_name])
    test_prediction = grid_search.predict(test)
    test_scores = (classification_report(test_labels, test_prediction)).split('\n')#TODO .2f -> .3f
    test_score =  ' '.join(test_scores[0].lstrip().split(' ')[:-1]) +'\n' + ' '.join(test_scores[-2].split(' ')[3:-1])
    train_prediction = grid_search.predict(train)
    train_accuracy = sum(train_prediction == train_labels)/len(train_labels)
    train_scores = (classification_report(train_labels, train_prediction)).split('\n')
    train_score =  ' '.join(train_scores[0].lstrip().split(' ')[:-1]) +'\n' + ' '.join(test_scores[-2].split(' ')[3:-1])
    test_accuracy = sum(test_prediction == test_labels)/len(test_labels)
    return 'cv %.3f test %.3f train %.3f' % (grid_search.best_score_, test_accuracy, train_accuracy) + '\n' + 'train: ' + train_score + '\n' + 'test: ' + test_score, k[:-1]

if __name__ == "__main__":
    """ Write evaluation results to the BagOfNgrams_Iclmdb_words_and_bigrams.csv DataFrame and to the output"""

    classifiers = ['SklearnLogReg', 'SklearnLinearSVC']#, 'SklearnMLP'
    best_params = ['best_parametersSklearnLogReg', 'best_parametersSklearnLinearSVC']
    search_parameters = dict()
    classifiers_dict=dict()

    search_parameters['SklearnLogReg'] = {'solver' : ('newton-cg', 'lbfgs', 'liblinear', 'sag'), 'penalty': ('l1', 'l2'), 'dual': (False, True), 'fit_intercept': (True, False), 'intercept_scaling': (1, 2, 3), 'max_iter': (100, 200, 400, 800, 1000), 'multi_class': ('ovr', 'multinomial')}
    #search_parameters['SklearnMLP'] = {'solver' : ('lbfgs', 'sgd', 'adam')}#TODO
    search_parameters['SklearnLinearSVC'] = {'loss' : ('hinge', 'squared_hinge'), 'penalty': ('l1', 'l2'), 'dual': (False, True), 'fit_intercept': (True, False), 'intercept_scaling': (1, 2, 3),  'max_iter': (100, 200, 400, 800, 1000), 'multi_class': ('ovr', 'crammer_singer')}

    classifiers_dict['SklearnLogReg'] = LogReg()
    #classifiers_dict['SklearnMLP'] = MLPClassifier(hidden_layer_sizes = (50, 50), max_iter=1000)
    classifiers_dict['SklearnLinearSVC'] = LinearSVC()

    alldocs = []  # will hold all docs in original order
    with open('aclImdb/alldata-id.txt', encoding='utf-8') as alldata:
        for line_no, line in enumerate(alldata):
            tokens = gensim.utils.to_unicode(line).split()
            words = tokens[1:]
            tags = [line_no] # `tags = [tokens[0]]` would also work at extra memory cost
            split = ['train','test','extra','extra'][line_no//25000]  # 25k train, 25k test, 25k extra
            sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
            alldocs.append(SentimentDocument(words, tags, split, sentiment))
    train_docs = [' '.join(doc.words) for doc in alldocs if doc.split == 'train']
    test_docs = [' '.join(doc.words) for doc in alldocs if doc.split == 'test']

    vectorizer = TfidfVectorizer(ngram_range = (1, 2))
    DocumentVectors0 = vectorizer.fit_transform(train_docs)

    DocumentVectors1 = vectorizer.transform(test_docs)

    y_1 = [1] * 12500
    y_0 = [0] * 12500

    df = pd.DataFrame(columns = classifiers+best_params)

    for classifier in classifiers:             
        accuracy, best = Classification(classifier, DocumentVectors0, y_1+y_0, DocumentVectors1, y_1+y_0)
        df.set_value(0, classifier, accuracy)
        df.set_value(0, 'best_parameters'+classifier, best)
        df.to_csv("BagOfNgrams_Iclmdb_words_and_bigrams.csv")
        print (classifier)
        print (accuracy)
    df.to_csv("BagOfNgrams_Iclmdb_words_and_bigrams.csv")
