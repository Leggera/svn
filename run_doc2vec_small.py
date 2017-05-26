import gensim
from gensim.models import Doc2Vec
#import gensim.models.doc2vec
import datetime
from contextlib import contextmanager
from timeit import default_timer
from scipy.special import expit
import numpy as np
import random
import pickle
from random import shuffle
import collections
import string
import os
import pandas as pd
import subprocess

def func(model, p_words, p_vec, N):

    train_error_value = 0
    if (len(p_words) < model.window):
        context_words = p_words
    else:    
        context_words = random.sample(p_words, model.window)
    c = []
    for word in context_words:
        c.append(model.wv.vocab[word])
    context_index = [c_.index for c_ in c]
    context_vectors = np.float64(model.wv.syn0)
    
    #retrive their vectors
    l1 = context_vectors[context_index]

    #and vector of the document itself
    
    
    word_indices = []
    while len(word_indices) < model.negative:
        #choose random word as negative samples
        w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
        word_indices.append(w)

    if (model.dm == 0):

        h = p_vec
        #maximize similarity between context and document and minimize for negative samples
        
        l2b = np.concatenate([l1, np.float64(model.syn1neg[word_indices, :])], axis = 0)
        
        #compute dot product between context and document with negative samples
        prod_term = np.dot(h, l2b.T).reshape(-1, 1)
        #compute cost function
        train_error_value -= np.sum(np.log(expit(-1 * prod_term[model.window:model.window + model.negative, :])), axis = 0)/N      
        
        train_error_value -= np.sum(np.log(expit(prod_term[:model.window, :])), axis = 0)/N
        

    elif (model.dm == 1):
        #maximize similarity between target and document with context and minimize for negative samples
        C = np.sum([p_vec, np.sum(l1[1:, :])])
        h = C/(l1.shape[0] + 1)
        
        l2b = np.concatenate([l1[0, :].reshape(1, -1), model.syn1neg[word_indices, :]], axis = 0)
        
        #compute dot product between target word and context, document, negative samples
        
        prod_term = np.dot(h, l2b.T).reshape(1, -1).T
        
        #compute cost function
        train_error_value -= np.sum(np.log(expit(-1 * prod_term[1:, :])), axis = 0)/N
        
        train_error_value -= np.sum(np.log(expit(prod_term[0, :])), axis = 0)/N
        
    return train_error_value

def cost(model, p, test_docs, N):
    
    train_error_value = 0
    for i, doc in zip(p, test_docs):
        p_vec = p[i][0].reshape(1, -1)
        #tag = p[i][1][0]
        #j = tag.find('SENT_')
        #p_id = int(tag[j+5:].split()[0])
        
        p_words = [word for word in doc.words if word in model.wv.vocab]
        
        if (len(p_words) > 0):
            train_error_value += func(model, p_words, p_vec, N)
    #print ('%d documents %f' % (N, np.sum(train_error_value)))
    return train_error_value

def cost_function(model, docs, N):

    train_error_value = 0
    
    #for each of N random documents
    for p_id in range(N):

        p_words = [word for word in docs[p_id].words if word in model.wv.vocab]
        p_vec = np.float64(model.docvecs[p_id].reshape(-1, 1).T)
        #choose some words from the document
        if (len(p_words) > 0):
            train_error_value += func(model, p_words, p_vec, N)
        
            
    #print ('%d documents %f' % (N, np.sum(train_error_value)))
    return train_error_value

def run_doc2vec(train_docs, dev_docs, test_docs, dm, size, window, alpha, negative, sample, cores, min_count, passes, output, diagnostics = False):

    assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

    model = Doc2Vec(dm=dm, size=size, window=window, alpha = alpha, min_alpha = alpha, negative=negative, sample=sample, workers=cores, min_count = min_count, iter = 1)
    model.build_vocab(train_docs)

    train_shuffled = train_docs
    
    whole_duration = 0
    
    if (diagnostics):
        infer_vecs = np.zeros((len(test_docs), size))
        dev_vecs = np.zeros((len(dev_docs), size))
        test_vectors = dict()
        dev_vectors = dict()
        neighb_num = 10
        words = []
        for doc in train_docs:
            words += doc.words 
        counter = collections.Counter(words)
        if not os.path.exists('diagnostics/'):
            os.mkdir('diagnostics/')
        i = output.find('/')
        diag_folder = 'diagnostics' + output[i:].replace(' ','_').replace('-','').replace('.txt', '')+'/'
        if not os.path.exists(diag_folder):
            os.mkdir(diag_folder)
        tmp_dir = 'temp' + output[:i].replace('.txt', '') + '/'
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        
        par_cols = ['dm', 'size', 'window', 'alpha', 'negative', 'sample', 'min_count', 'epochs', 'cores']
        par_use = [dm, size, window, alpha, negative, sample, min_count, passes, cores]
        par_df = pd.DataFrame(columns=par_cols)        
        for par_c, par_u in zip(par_cols, par_use):
            par_df.loc[0, par_c] = par_u
        par_df.to_csv(diag_folder+'pars.csv')

        df = pd.DataFrame(columns=['neighbours'])

        p_ids = np.linspace(0, len(train_docs) - 1, num = 5)

        dev = np.zeros(passes+1)
        train = np.zeros(passes+1)
        train_N = 30
        train_ids_for_cost = np.linspace(0, len(train_docs) - 1, num = train_N)
        train_for_cost = []
        for i in train_ids_for_cost:
            train_for_cost += [train_docs[int(i)]]

    

    print("START %s" % datetime.datetime.now())

    with elapsed_timer() as elapsed:

         

        for epoch in range(passes):

            if (diagnostics):
                if (epoch == 0):
                    diagnose(diag_folder, model, counter, p_ids, neighb_num, df, dev, train, epoch, alpha, passes, train_for_cost, train_N, dev_docs, dev_vectors, dev_vecs, output)
            shuffle(train_shuffled)            
            model.train(train_shuffled, total_examples = len(train_docs), epochs = 1)
            #model.alpha = model.alpha / 2
            #model.min_alpha = model.alpha
            print ('epoch %d' % (epoch + 1))
            #N = 1000
            if (diagnostics):
                diagnose(diag_folder, model, counter, p_ids, neighb_num, df, dev, train, epoch+1, alpha, passes, train_for_cost, train_N, dev_docs, dev_vectors, dev_vecs, output)

        for i, doc in enumerate(test_docs):
            infer_vecs[i, :] = model.infer_vector(doc.words, alpha=alpha, min_alpha=alpha, steps=passes)
            test_vectors[i] =  tuple([infer_vecs[i, :], doc.tags])
        
        test = cost(model, test_vectors, test_docs, len(test_docs))
        #print (test)

    whole_duration += elapsed()
    model.save(output)
    f = open(output + 'test', 'wb')
    pickle.dump(test_vectors, f)
    if (diagnostics):
        dev_pickle = open(diag_folder+'dev.npy', 'wb')
        pickle.dump(dev, dev_pickle)
        train_pickle = open(diag_folder+'train.npy', 'wb')
        pickle.dump(train, train_pickle)
        print ('dev_cost (%d documents)' %len(dev_docs), dev)
        print ('train_cost', train)
    print ('infer_cost', test)

    print("END %s" % str(datetime.datetime.now()))
    print("duration %s" % str(whole_duration))

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def diagnose(diag_folder, model, counter, p_ids, neighb_num, df, dev, train, epoch, alpha, passes, train_for_cost, train_N, dev_docs, dev_vectors, dev_vecs, output):
    for (word, count) in (counter.most_common()[165:195]):
        if (word not in string.punctuation):     
            
            for k, g in enumerate(model.wv.most_similar(word, topn=neighb_num)):
                n = []
                n += ('%s ' % g[0])
                n += ('%f' % g[1])
                df.loc[word + '_'+ str(epoch), str(k+1)] = ''.join(n)
            
    for p_id in p_ids:
        
        for k, g in enumerate(model.wv.similar_by_vector(model.docvecs[int(p_id)], topn=neighb_num)):
            n = []                        
            n += ('%s ' % g[0])
            n += ('%f' % g[1])
            df.loc[str(int(p_id))+'_'+str(epoch), str(k+1)] = ''.join(n)

    for i, doc in enumerate(dev_docs):
        dev_vecs[i, :] = model.infer_vector(doc.words, alpha=alpha, min_alpha=alpha, steps=passes)
        dev_vectors[i] = tuple([dev_vecs[i, :], doc.tags])
    dev[epoch] = cost(model, dev_vectors, dev_docs, len(dev_docs))
    train[epoch] = cost_function(model, train_for_cost, train_N)
    model.save('temp' + output)
    tmp_f = open('temp' + output + 'test', 'wb')
    pickle.dump(dev_vectors, tmp_f)
    i = output.find('/')
    subprocess.call(['python3', 'IMDB_small_dataframe.py', 'IMDB' , 'temp' + output[:i+1] , output[i+1:], diag_folder, str(epoch)])
    
    print (dev[epoch])
    print (train[epoch])
    sort_df = df.sort_index()
    sort_df.to_csv(diag_folder+'neighbours.csv')
