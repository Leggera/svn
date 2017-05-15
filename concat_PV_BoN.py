from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import sys
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression as LogReg
#from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import gensim
from collections import namedtuple
import os
import pickle
import numpy as np
import scipy
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

def main(corpora, p2v_dir):
    

    BoN_models = [TfidfVectorizer(ngram_range = (1, 1)), TfidfVectorizer(ngram_range = (1, 2)), TfidfVectorizer(ngram_range = (1, 3))]

    SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

    if ('IMDB' in corpora):
        alldocs = []  # will hold all docs in original order
        with open('alldata-id.txt', encoding='utf-8') as alldata:
            for line_no, line in enumerate(alldata):
                tokens = gensim.utils.to_unicode(line).split()
                words = tokens[1:]
                tags = [line_no] # `tags = [tokens[0]]` would also work at extra memory cost
                split = ['train','test','extra','extra'][line_no//25000]  # 25k train, 25k test, 25k extra
                sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
                alldocs.append(SentimentDocument(words, tags, split, sentiment))
        train_docs = [' '.join(doc.words) for doc in alldocs if doc.split == 'train']
        test_docs = [' '.join(doc.words) for doc in alldocs if doc.split == 'test']
        y_1 = [1] * 12500
        y_0 = [0] * 12500
        train_labels = y_1 + y_0
        test_labels = y_1 + y_0

    elif ('20ng' in corpora):
        train_docs = newsgroups_train.data
        test_docs = newsgroups_test.data
        
    index = 0

    for p2v_file in os.listdir(p2v_dir): #for every model in the vectors directory
        if p2v_file.endswith('.txt'):
            for BoN_model in BoN_models:
                index += 1

                for column in parameters:  
                    i = p2v_file.find(column)
                    if (i != -1):
                        value = p2v_file[i:].split()[1]
                        df.set_value(index, column, value)
                    else:
                        df.set_value(index, column, default_parameters[column])
                
                
                p2v_model = Doc2Vec.load(p2v_dir + p2v_file)
                f = open(p2v_dir + p2v_file + 'test', 'rb')
                p = pickle.load(f)
                if ('IMDB' in corpora):
                    p2v_DocumentVectors0 = np.array([p2v_model.docvecs['SENT_'+str(i)] for i in range(0, 25000)])
                else:
                    p2v_DocumentVectors0 = np.array([p2v_model.docvecs[tag] for tag in p2v_model.docvecs.doctags if 'train' in tag])
                    test_labels = [p[i][1][0].split()[2] for i in p]
                    train_labels = [tag.split()[2] for tag in model_d2v.docvecs.doctags if 'train' in tag]
                BoN_DocumentVectors0 = BoN_model.fit_transform(train_docs)
                
                DocumentVectors0 = scipy.sparse.hstack([p2v_DocumentVectors0, BoN_DocumentVectors0])
                p2v_DocumentVectors1 = np.concatenate([p[i][0].reshape(1, -1) for i in p])
                BoN_DocumentVectors1 = BoN_model.transform(test_docs)
                DocumentVectors1 = scipy.sparse.hstack([p2v_DocumentVectors1, BoN_DocumentVectors1])
                print (p2v_file)
                standard_scaler = StandardScaler(with_mean = False)
                X_train = standard_scaler.fit_transform(DocumentVectors0)
                X_test = standard_scaler.transform(DocumentVectors1)
                for classifier in classifiers:
                    accuracy, best = Classification(classifier, X_train, train_labels, X_test, test_labels)
                    #write it all into DataFrame
                    df.set_value(index, classifier, accuracy)
                    df.set_value(index, 'best_parameters' + classifier, best)
                    df.to_csv("Res_concat_PV_BoN.csv")
                    print (accuracy)
                for classifier in classifiers:
                    accuracy, best = Classification(classifier, scale(DocumentVectors0, with_mean=False), train_labels, scale(DocumentVectors1, with_mean=False), test_labels)
                    #write it all into DataFrame
                    print (accuracy)
                
                #and to the output
                                
                

def Classification(classifier, train, train_labels, test, test_labels):
    
    """ Train and evaluate classifier """
    
    k = ""
    t0 = time() #start the clock
    #GridSearch
    clf = GridSearchCV(classifiers_dict[classifier], param_grid = search_parameters[classifier], error_score=0.0, n_jobs = 1)
    clf.fit(train, train_labels)
    best_parameters = clf.best_estimator_.get_params()#get parameters that worked best on cross-validation
    
    for param_name in sorted(search_parameters[classifier].keys()):
        k += "%s: %r\n" % (param_name, best_parameters[param_name]) + "cv %.3f " % clf.best_score_ #write it all in one string with cv score

    print("done in %0.3fs" % (time() - t0)) #stop the clock
    
    test_prediction = clf.predict(test) #predict on test
    test_accuracy = sum(test_prediction == test_labels)/len(test_labels) #test accuracy
    test_scores = (classification_report(test_labels, test_prediction)).split('\n') #precision, recall and F-score on test data
    test_score =  ' '.join(test_scores[0].lstrip().split(' ')[:-1]) +'\n' + ' '.join(test_scores[-2].split(' ')[3:-1])
    train_prediction = clf.predict(train) #predict on train
    train_accuracy = sum(train_prediction == train_labels)/len(train_labels) #train accuracy
    train_scores = (classification_report(train_labels, train_prediction)).split('\n')#precision, recall and F-score on train data
    train_score =  ' '.join(train_scores[0].lstrip().split(' ')[:-1]) +'\n' + ' '.join(train_scores[-2].split(' ')[3:-1])
    return 'test %.3f train %.3f' % (test_accuracy, train_accuracy) + '\n' + 'train: ' + train_score + '\n' + 'test: ' + test_score, k[:-1]

if __name__ == "__main__":
    classifiers_dict = dict()
    search_parameters = dict()
    default_parameters = dict()
    classifiers_dict['LogReg'] = LogReg()
    classifiers_dict['LinearSVC'] = LinearSVC()
    search_parameters['LogReg'] = {'C': (10**-5, 3*10**-5, 10**-4, 3*10**-4, 10**-3, 3*10**-3,10**-2, 3*10**-2,10**-1, 3*10**-1, 1)}
    search_parameters['LinearSVC'] = {'C': (10**-5, 3*10**-5, 10**-4, 3*10**-4, 10**-3, 3*10**-3,10**-2, 3*10**-2,10**-1, 3*10**-1, 1)}

    d0 = ['implementation']
    columns = ['cbow', 'size', 'alpha', 'window', 'negative', 'sample', 'min_count']
    best_params = ['best_parametersLogReg', 'best_parametersLinearSVC']
    classifiers = ['LogReg', 'LinearSVC']
    df = pd.DataFrame(columns = d0 + columns + classifiers + best_params)
    parameters = ['size', 'alpha', 'window', 'negative', 'min_count']
    default_parameters['size'] = 150
    default_parameters['alpha'] = 0.05
    default_parameters['window'] = 10
    default_parameters['negative'] = 25
    default_parameters['min_count'] = 1
    main(sys.argv[1], sys.argv[2])
