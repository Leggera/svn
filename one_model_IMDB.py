import pandas as pd
import gensim
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression as LogReg
#from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from time import time
import os
from sklearn.metrics import classification_report
import numpy as np
#import statsmodels.api as sm
import argparse
from datetime import datetime
import sys
import traceback
import pickle

def DocumentVectors(model, model_name):

    """ Load paragraph2vec vectors"""
    if (model_name == "word2vec_c"):
        model_w2v = gensim.models.Doc2Vec.load_word2vec_format(model , binary=False)
        vec_vocab = [w for w in model_w2v.vocab if "_*" in w]
        vec_vocab = sorted(vec_vocab, key = lambda x: int(x[2:]))
        DocumentVectors0 = [model_w2v[w] for w in vec_vocab[:25000]]
        DocumentVectors1 = [model_w2v[w] for w in vec_vocab[25000:50000]]
    elif(model_name == "doc2vec"): #TODO
        model_d2v = Doc2Vec.load(model)    #loading saved model 
        DocumentVectors0 = np.array([model_d2v.docvecs['SENT_'+str(i)] for i in range(0, 25000)]) #first 25000 are labeled train data
        f = open(model + 'test', 'rb')
        p = pickle.load(f)
        DocumentVectors1 = np.concatenate([p[i][0].reshape(1, -1) for i in p])
        #DocumentVectors1 = [model_d2v.docvecs['SENT_'+str(i)] for i in range(25000, 50000)]#second 25000 are labeled test data
        
    return (DocumentVectors0, DocumentVectors1)

def Classification(classifier, C, train, train_labels, test, test_labels):
    
    """ Train and evaluate classifier """
    
    k = ""
    t0 = time() #start the clock
    if (C is not None): #just Classification
        clf = classifiers_dict[classifier]
        clf.fit(train, train_labels)
        k = "C " + str(C)
    else:               #GridSearch
        clf = GridSearchCV(classifiers_dict[classifier], param_grid = search_parameters[classifier], error_score=0.0, n_jobs = 30)
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

def time_str():

    """return current date and time with replaced spaces to '_'"""

    return ('%s' % datetime.now().replace(microsecond=0)).replace(' ','_')


def main(space_dir, classifier, C = None):

    """ Write evaluation results to the Res_concat_IMDB.csv DataFrame in separate directories and to the output"""
    
    #future DataFrame fields
    d0 = ['implementation']
    parameters = ['size', 'alpha', 'window', 'negative', 'min_count']
    columns = ['size', 'alpha', 'window', 'negative', 'cbow0', 'sample', 'min_count']
    best_params = ['best_parameters']
    classifiers = ['LogReg', 'LinearSVC']
    
    #default parameters from the article
    default_parameters['size'] = 150
    default_parameters['alpha'] = 0.05
    default_parameters['window'] = 10
    default_parameters['negative'] = 25
    default_parameters['min_count'] = 1

    if (C is not None): #if C was given as an input value then initialize classifier with it
        classifiers_dict['LogReg'] = LogReg(C = C)
        #classifiers_dict['SklearnMLP'] = MLPClassifier(hidden_layer_sizes = (50, 50), max_iter=1000)
        classifiers_dict['LinearSVC'] = LinearSVC(C = C)
        #classifiers_dict['StatModelsLogReg'] = sm.Logit()
    else: #else prepare for GridSerach
        classifiers_dict['LogReg'] = LogReg()
        classifiers_dict['LinearSVC'] = LinearSVC()
        search_parameters['LogReg'] = {'C': (10**-5, 3*10**-5, 10**-4, 3*10**-4, 10**-3, 3*10**-3,10**-2, 3*10**-2,10**-1, 3*10**-1, 1), 'max_iter': (200, 400, 1000, 2000)}
        search_parameters['LinearSVC'] = {'C': (10**-5, 3*10**-5, 10**-4, 3*10**-4, 10**-3, 3*10**-3,10**-2, 3*10**-2,10**-1, 3*10**-1, 1), 'max_iter': (200, 400, 1000, 2000)}
    
    
    
    
    #index  = 0
    

    for other_model in os.listdir(space_dir): #for every model in the vectors directory
        if other_model.endswith('.txt'):      #if it is the name of saved model
            
              df= pd.DataFrame(columns = d0+columns + classifiers +best_params)#initialize DataFrame
              
              
              string = other_model.split(".txt")[0] #name of the PV-DM model

              #index += 1
              index = 1 #only one string int the DataFrame                          

              #putting parameters into DataFrame  
              for column in parameters:  
                  i = string.find(column)

                  if (i != -1):
                      value = string[i:].split()[1]
                      df.set_value(index, column, value)
                  else:
                      df.set_value(index, column, default_parameters[column])

              i = string.find('sample')
              if (i != -1):
                  value = string[i:].split()[1]
                  df.set_value(index, 'sample', value)
                  if ('cbow 0' in other_model): 
                    df.set_value(index, 'cbow', '0')
                  else:
                    df.set_value(index, 'cbow', '1')
              else:
                  if ('cbow 0' in other_model): 
                    df.set_value(index, 'cbow', '0')
                    df.set_value(index, 'sample', '1e-2')
                  else:
                    df.set_value(index, 'cbow', '1')
                    df.set_value(index, 'sample', '1e-4')

                  
              
              #load train and test vectors from PV-DM model + labels
              try:
                  DocumentVectors0, DocumentVectors1 = DocumentVectors(space_dir + other_model+'.txt', implementation)
              except:
                  print (other_model)#print which model causes the problem
                  traceback.print_exc(file=sys.stdout)
                  continue

              y_1 = [1] * 12500
              y_0 = [0] * 12500
              train_labels = y_1 + y_0
              test_labels = y_1 + y_0
              dir_name = (other_model + ''.join(samples)).replace(' ','_').replace('-','')#name directory after model parameters
              run_dir = './runs_IMDB_one/%s-%s/' % (dir_name, time_str())#and after starting time
              os.makedirs(run_dir, exist_ok=True) #make this directory
              #get accuracy, precision, recall, etc. and best parameters (if C was in input then it will be chosen as the best par)
              accuracy, best = Classification(classifier, C, DocumentVectors0, train_labels, DocumentVectors1, test_labels)
              #write it all into DataFrame
              df.set_value(index, classifier, accuracy)
              df.set_value(index, 'best_parameters', best)
              df.to_csv(run_dir + "Res_one_IMDB" + classifier + ".csv")
              #and to the output
              print (other_model)
              print (accuracy)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input: vector directory, classifier name, regularization parameter (optional). Writes to stdout the best classifier accuracy, precision, recall, F-score and in case of no given C parameter - cross validation accuracy from GridSearch. Also writes it all to the DataFrame depending on parameters and dataset')
    
    parser.add_argument("-vectors", nargs='?', default='space_p2v/', help = 'paragraph2vec vectors directory')
    
    parser.add_argument("-classifier", choices=['lr','linearsvc'], help = 'classifier name (lr or linearsvc)')
    
    parser.add_argument("-C", nargs='?', default='None', help = 'regularization parameter')
    
    args = parser.parse_args()
    space_dir = args.vectors


    if (space_dir is not None):
        if (not space_dir.endswith('/')):
            space_dir = space_dir + '/'

    if (args.classifier == 'lr'):
        classifier = 'LogReg'
    elif (args.classifier == 'linearsvc'):
        classifier = 'LinearSVC'
    C = args.C
    if (C is not 'None'):
        C = float(C)
    else:
        C = None
    default_parameters = dict()
    classifiers_dict = dict()
    search_parameters = dict()

    main(space_dir, classifier, C)
