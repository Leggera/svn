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

def main(corpora, p2v_dir, p2v_file, diag_dir, epoch):
    

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
        
    elif ('20ng' in corpora):
        train_docs = newsgroups_train.data
        test_docs = newsgroups_test.data
        
           

    for column in parameters:  
        i = p2v_file.find(column)
        if (i != -1):
            value = p2v_file[i:].split()[1]
            df.set_value(epoch, column, value)
        else:
            df.set_value(epoch, column, default_parameters[column])
    
    
    p2v_model = Doc2Vec.load(p2v_dir + p2v_file)
    f = open(p2v_dir + p2v_file + 'test', 'rb')
    p = pickle.load(f)
    if ('IMDB' in corpora):
        dev = 140
        p2v_DocumentVectors0 = np.array([p2v_model.docvecs['SENT_'+str(i)] for i in range(12425, 12500 - dev)] + [p2v_model.docvecs['SENT_'+str(i)] for i in range(12500 + dev, 12575)])
        y_1 = [1] * (475 - dev)
        y_0 = [0] * (475 - dev)
        train_labels = y_1 + y_0

        test_labels = [1] * dev + [0] * dev            
    else:
        p2v_DocumentVectors0 = np.array([p2v_model.docvecs[tag] for tag in p2v_model.docvecs.doctags if 'train' in tag])
        test_labels = [p[i][1][0].split()[2] for i in p]
        train_labels = [tag.split()[2] for tag in model_d2v.docvecs.doctags if 'train' in tag]
    
    
    p2v_DocumentVectors1 = np.concatenate([p[i][0].reshape(1, -1) for i in p])
    

    for classifier in classifiers:
        accuracy, best = Classification(classifier, p2v_DocumentVectors0, train_labels, p2v_DocumentVectors1, test_labels)
        #write it all into DataFrame
        df.set_value(epoch, classifier, accuracy)
        df.set_value(epoch, 'best_parameters' + classifier, best)
        df.set_value(epoch, 'epoch', epoch)
        df.to_csv(diag_dir+"Res_PV_IMDB.csv")
        print (accuracy)
    
            
                                
                

def Classification(classifier, train, train_labels, test, test_labels):
    
    """ Train and evaluate classifier """
    
    k = ""
    t0 = time() #start the clock
    #GridSearch
    clf = GridSearchCV(classifiers_dict[classifier], cv = 3, param_grid = search_parameters[classifier], error_score=0.0, n_jobs = 3)
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
    search_parameters['LogReg'] = {'C': (3*10**-3, 3*10**-2, 3*10**-1)}
    search_parameters['LinearSVC'] = {'C': (3*10**-3, 3*10**-2, 3*10**-1)}

    d0 = ['implementation', 'epoch']
    columns = ['cbow', 'size', 'alpha', 'window', 'negative', 'sample', 'min_count']
    best_params = ['best_parametersLogReg', 'best_parametersLinearSVC']
    classifiers = ['LogReg', 'LinearSVC']
    diag_dir = sys.argv[4]
    epoch = int(sys.argv[5])
    if (epoch == 0):
        df = pd.DataFrame(columns = d0 + columns + classifiers + best_params)
    else:
        df = pd.DataFrame.from_csv(diag_dir+"Res_PV_IMDB.csv")
    parameters = ['size', 'alpha', 'window', 'negative', 'min_count']
    default_parameters['size'] = 150
    default_parameters['alpha'] = 0.05
    default_parameters['window'] = 10
    default_parameters['negative'] = 25
    default_parameters['min_count'] = 1
    
    main(sys.argv[1], sys.argv[2], sys.argv[3], diag_dir, epoch)
