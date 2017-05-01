
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
    t0 = time() #start the clock
    grid_search.fit(train, train_labels)#GridSearch
    print("done in %0.3fs" % (time() - t0))#stop the clock
    best_parameters = grid_search.best_estimator_.get_params()#get parameters that worked best on cross-validation
    k = ""
    for param_name in sorted(search_parameters[classifier].keys()):
        k += "%s: %r\n" % (param_name, best_parameters[param_name])#write it all in one string
    test_prediction = grid_search.predict(test)#predict on test
    test_scores = (classification_report(test_labels, test_prediction)).split('\n')#precision, recall and F-score on test data
    test_score =  ' '.join(test_scores[0].lstrip().split(' ')[:-1]) +'\n' + ' '.join(test_scores[-2].split(' ')[3:-1])
    train_prediction = grid_search.predict(train)#predict on train
    train_accuracy = sum(train_prediction == train_labels)/len(train_labels)#train accuracy
    train_scores = (classification_report(train_labels, train_prediction)).split('\n')#precision, recall and F-score on train data
    train_score =  ' '.join(train_scores[0].lstrip().split(' ')[:-1]) +'\n' + ' '.join(test_scores[-2].split(' ')[3:-1])
    test_accuracy = sum(test_prediction == test_labels)/len(test_labels)#test accuracy
    return 'cv %.3f test %.3f train %.3f' % (grid_search.best_score_, test_accuracy, train_accuracy) + '\n' + 'train: ' + train_score + '\n' + 'test: ' + test_score, k[:-1]
    

if __name__ == "__main__":
    """ Write evaluation results to the BagOfNgrams_20ng.csv DataFrame and to the output"""

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

    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    vectorizer = TfidfVectorizer()
    DocumentVectors0 = vectorizer.fit_transform(newsgroups_train.data)#vectorize train data

    DocumentVectors1 = vectorizer.transform(newsgroups_test.data)#vectorize test data

    df = pd.DataFrame(columns = classifiers+best_params)

    for classifier in classifiers:
        #gridSearch and evaluation
        accuracy, best = Classification(classifier, DocumentVectors0, newsgroups_train.target, DocumentVectors1, newsgroups_test.target)
        #writing to dataframe
        df.set_value(0, classifier, accuracy)
        df.set_value(0, 'best_parameters'+classifier, best)
        print (classifier)
        print (accuracy)
    df.to_csv("BagOfNgrams_20ng.csv")#saving to csv
