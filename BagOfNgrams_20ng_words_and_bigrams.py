
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LogReg
#from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from time import time
from sklearn.metrics import classification_report

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
    """ Write evaluation results to the BagOfNgrams_20ng_words_and_bigrams.csv DataFrame and to the output"""

    classifiers = ['SklearnLogReg', 'SklearnLinearSVC']#, 'SklearnMLP'
    best_params = ['best_parametersSklearnLogReg', 'best_parametersSklearnLinearSVC']
    search_parameters = dict()
    classifiers_dict=dict()

    search_parameters['SklearnLogReg'] = {'C': (10**-5, 3*10**-5, 10**-4, 3*10**-4, 10**-3, 3*10**-3,10**-2, 3*10**-2,10**-1, 3*10**-1, 1),  'max_iter': (100, 200, 400, 800, 1000)}
    search_parameters['SklearnLinearSVC'] = {'C': (10**-5, 3*10**-5, 10**-4, 3*10**-4, 10**-3, 3*10**-3,10**-2, 3*10**-2,10**-1, 3*10**-1, 1),  'max_iter': (100, 200, 400, 800, 1000)}
    #search_parameters['SklearnLogReg'] = {'solver' : ('newton-cg', 'lbfgs', 'liblinear', 'sag'), 'penalty': ('l1', 'l2'), 'dual': (False, True), 'fit_intercept': (True, False), 'intercept_scaling': (1, 2, 3), 'max_iter': (100, 200, 400, 800, 1000), 'multi_class': ('ovr', 'multinomial')}
    #search_parameters['SklearnMLP'] = {'solver' : ('lbfgs', 'sgd', 'adam')}#TODO
    #search_parameters['SklearnLinearSVC'] = {'loss' : ('hinge', 'squared_hinge'), 'penalty': ('l1', 'l2'), 'dual': (False, True), 'fit_intercept': (True, False), 'intercept_scaling': (1, 2, 3),  'max_iter': (100, 200, 400, 800, 1000), 'multi_class': ('ovr', 'crammer_singer')}

    classifiers_dict['SklearnLogReg'] = LogReg()
    #classifiers_dict['SklearnMLP'] = MLPClassifier(hidden_layer_sizes = (50, 50), max_iter=1000)
    classifiers_dict['SklearnLinearSVC'] = LinearSVC()

    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    vectorizer = TfidfVectorizer(ngram_range = (1, 2))
    DocumentVectors0 = vectorizer.fit_transform(newsgroups_train.data)

    DocumentVectors1 = vectorizer.transform(newsgroups_test.data)

    df = pd.DataFrame(columns = classifiers+best_params)

    for classifier in classifiers:             
        accuracy, best = Classification(classifier, DocumentVectors0, newsgroups_train.target, DocumentVectors1, newsgroups_test.target)
        df.set_value(0, classifier, accuracy)
        df.set_value(0, 'best_parameters'+classifier, best)
        print (classifier)
        print (accuracy)
    df.to_csv("BagOfNgrams_20ng_words_and_bigrams.csv")
