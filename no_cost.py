import gensim
from gensim.models import Doc2Vec
#import gensim.models.doc2vec
import datetime
from contextlib import contextmanager
from timeit import default_timer
import numpy as np
import random
import pickle
from random import shuffle
import string
import os

def run_doc2vec(train_docs, dev_docs, test_docs, dm, size, window, alpha, negative, sample, cores, min_count, passes, output):

    assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

    model = Doc2Vec(dm=dm, dbow_words=1, size=size, window=window, alpha = alpha, negative=negative, sample=sample, workers=cores, min_count = min_count, iter = 1)
    model.build_vocab(train_docs + dev_docs)


    
    infer_vecs = np.zeros((len(test_docs), size))
    test_vectors = dict()
    train_shuffled = train_docs
    whole_duration = 0

    print("START %s" % datetime.datetime.now())

    with elapsed_timer() as elapsed:

        for epoch in range(passes):

            shuffle(train_shuffled)
            
            model.train(train_shuffled, total_examples = len(train_docs), epochs = 1)
            
            print ('epoch %d' % (epoch + 1))

  
        for i, doc in enumerate(test_docs):
            infer_vecs[i, :] = model.infer_vector(doc.words, alpha=alpha, min_alpha=0.0001, steps=25)
            test_vectors[i] =  tuple([infer_vecs[i, :], doc.tags])
        

    whole_duration += elapsed()
    model.save(output)
    f = open(output + 'test', 'wb')
    pickle.dump(test_vectors, f)

    print("END %s" % str(datetime.datetime.now()))
    print("duration %s" % str(whole_duration))

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start
