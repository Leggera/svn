import gensim
from gensim.models import Doc2Vec
import gensim.models.doc2vec
import datetime
from contextlib import contextmanager
from timeit import default_timer

def run_doc2vec(train_docs, test_docs, alldocs, dm, size, window, alpha, negative, sample, cores, min_count, passes, output):

    assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

    model = Doc2Vec(dm=dm, size=size, window=window, alpha = alpha, negative=negative, sample=sample, workers=cores, min_count = min_count, iter=passes)
    model.build_vocab(alldocs)

    #min_alpha = 0.001
    #alpha_delta = (alpha - min_alpha) / passes

    print("START %s" % datetime.datetime.now())

    
    whole_duration = 0

    duration = 'na'
    with elapsed_timer() as elapsed:
        model.train(train_docs, total_examples = len(train_docs), epochs = model.iter)
        duration = '%.1f' % elapsed()
        whole_duration += elapsed() 

        model.train_words = False
        model.train_labels = True
        model.train(test_docs, total_examples = len(test_docs), epochs = model.iter)
    model.save(output)
    print("END %s" % str(datetime.datetime.now()))
    print("duration %s" % str(whole_duration))

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start
