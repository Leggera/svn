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

def func(model, p_words, p_id, N):

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
    p_vec = np.float64(model.docvecs[p_id].reshape(-1, 1).T)
    
    word_indices = []
    while len(word_indices) < model.negative:
        #choose random word as negative samples
        w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
        word_indices.append(w)

    if (model.dm == 0):

        h = p_vec
        #maximize similarity between context and document and minimize for negative samples
        l2b = np.concatenate([l1, np.float64(model.syn1neg[word_indices])], axis = 0)
        #compute dot product between context and document with negative samples
        prod_term = np.dot(h, l2b.T).reshape(-1, 1)
        #compute cost function
        train_error_value -= np.sum(np.log(expit(-1 * prod_term[model.window:model.window + model.negative, :])), axis = 0)/N      
        
        train_error_value -= np.sum(np.log(expit(prod_term[:model.window, :])), axis = 0)/N

    elif (model.dm == 1):
        #maximize similarity between target and document with context and minimize for negative samples
        C = np.sum([p_vec, np.sum(l1[1:, :])])
        h = C/(l1.shape[0] + 1)
        l2b = np.concatenate([l1[0, :].reshape(1, -1), model.syn1neg[word_indices]], axis = 0)
        #compute dot product between target word and context, document, negative samples
        
        prod_term = np.dot(h, l2b.T).reshape(1, -1).T
        
        #compute cost function
        train_error_value -= np.sum(np.log(expit(-1 * prod_term[1:, :])), axis = 0)/N
        
        train_error_value -= np.sum(np.log(expit(prod_term[0, :])), axis = 0)/N
        
    return train_error_value

def cost(model, p, test_docs, N):
    
    train_error_value = 0
    for i in p:
        p_vec = p[i][0].reshape(1, -1)
        tag = p[i][1][0]
        p_id = int(tag[5:])
        p_words = [word for word in test_docs[p_id - 25000].words if word in model.wv.vocab]#TODO 25000 = len(train_docs)
        train_error_value += func(model, p_words, p_id, N)
    #print ('%d documents %f' % (N, np.sum(train_error_value)))
    return train_error_value

def cost_function(model, docs, N):

    train_error_value = 0
    
    #for each of N random documents
    for p_id in random.sample(range(len(docs)), N):

        p_words = [word for word in docs[p_id].words if word in model.wv.vocab]
        #choose some words from the document
        train_error_value += func(model, p_words, p_id, N)
            
    #print ('%d documents %f' % (N, np.sum(train_error_value)))
    return train_error_value

def run_doc2vec(train_docs, dev_docs, test_docs, dm, size, window, alpha, negative, sample, cores, min_count, passes, output):

    assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

    model = Doc2Vec(dm=dm, dbow_words=1, size=size, window=window, alpha = alpha, negative=negative, sample=sample, workers=cores, min_count = min_count, iter = 1)
    model.build_vocab(train_docs + dev_docs)


    print("START %s" % datetime.datetime.now())

    train_shuffled = train_docs
    infer_vecs = np.zeros((len(test_docs), size))
    test_vectors = dict()
    whole_duration = 0
    
    words = []
    for doc in train_docs:
        words += doc.words 
    counter = collections.Counter(words)
    
           
    
    i = output.find('/')
    if not os.path.exists('neighbours_' + output[:i] + '/'):
        os.mkdir('neighbours_' + output[:i]      + '/')
    
    n_dir = 'neighbours_' + output + '.csv'

    df = pd.DataFrame(index=['word'], columns=['epoch'])

    p_ids = np.linspace(0, len(train_docs) - 1, num = 5)

    dev = np.zeros(passes)
    train = np.zeros(passes)
    duration = 'na'

    with elapsed_timer() as elapsed:

        '''min_alpha = 0.0001
        if (passes > 1):
            alpha_delta = (alpha - min_alpha) / (passes - 1)
        else:
            alpha_delta = 0'''      

        for epoch in range(passes):

            shuffle(train_shuffled)
            
            model.train(train_shuffled, total_examples = len(train_docs), epochs = 1)
            for (word, count) in (counter.most_common()[165:195]):
                if (word not in string.punctuation):     
                    n = []
                    for g in model.wv.most_similar(word, topn=15):
                        n += ('%s ' % g[0])
                        n += ('%f\n' % g[1])
                    df.loc[word, epoch+1] = ''.join(n)
                    
            for p_id in p_ids:
                n = []
                for g in model.wv.similar_by_vector(model.docvecs[int(p_id)], topn=15):
                    n += ('%s ' % g[0])
                    n += ('%f\n' % g[1])
                df.loc[p_id, epoch+1] = ''.join(n)
            #model.alpha -= alpha_delta
            print ('epoch %d' % (epoch + 1))
            #N = 1000
            dev[epoch] = cost_function(model, dev_docs, len(dev_docs))
            train[epoch] = cost_function(model, train_docs, len(train_docs))
            print (dev[epoch])
            print (train[epoch])
        df.to_csv(n_dir)

        duration = '%.1f' % elapsed()

        whole_duration += elapsed()
        
        for i, doc in enumerate(test_docs):
            infer_vecs[i, :] = model.infer_vector(doc.words, alpha=alpha, min_alpha=0.0001, steps=25)
            test_vectors[i] =  tuple([infer_vecs[i, :], doc.tags])
        test = cost(model, test_vectors, test_docs, len(test_docs))
        print (test)

    model.save(output)
    f = open(output + 'test', 'wb')
    pickle.dump(test_vectors, f)

    print ('dev_cost (%d documents)' %N, dev)
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
