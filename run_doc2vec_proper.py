import gensim
from collections import namedtuple
from run_doc2vec import run_doc2vec
import argparse

def load_data():

    SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

    alldocs = []  # will hold all docs in original order

    with open(corpora, 'r') as alldata:
        for line_no, line in enumerate(alldata):
            tokens = gensim.utils.to_unicode(line).split()
            words = tokens[1:]
            tags = ['SENT_'+ str(line_no)] # `tags = [tokens[0]]` would also work at extra memory cost
            if (line_no < 25000):
                split = 'train'
                sentiment = [1.0, 0.0][line_no//12500]
            elif (line_no < 50000):
                split = 'test'
                sentiment = [1.0, 0.0, 1.0, 0.0][line_no//12500]#
            else:
                split = 'extra'
                sentiment = None
             # [12.5K pos, 12.5K neg]*2 then unknown
            alldocs.append(SentimentDocument(words, tags, split, sentiment))

    train_docs = [doc for doc in alldocs if (doc.split == 'train' or doc.split == 'extra')]
    test_docs = [doc for doc in alldocs if doc.split == 'test']
    return train_docs[:-10000], train_docs[-10000:], test_docs, alldocs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-cbow", nargs='?', default='1', help = '1 for PV-DM, 0 for PV-DBOW')
    parser.add_argument("-size", nargs='?', default='150', help = 'vector size')
    parser.add_argument("-window", nargs='?', default='10', help = 'context size')
    parser.add_argument("-negative", nargs='?', default='25', help = 'number of negative samples')
    parser.add_argument("-sample", nargs='?', default='1e-4', help = 'downsampling')
    parser.add_argument("-iter", nargs='?', default='25', help = 'number of epochs')
    parser.add_argument("-alpha", nargs='?', default='0.05', help = 'initial learning rate')
    parser.add_argument("-output", help = 'file to save trained model to')
    parser.add_argument("-threads", nargs='?', default='4', help = 'number of parallel processes')
    parser.add_argument("-min_count", nargs='?', default='5', help = 'minimal word frequency threshold')

    parser.add_argument("-train", help = 'path to the corpora')

    args = parser.parse_args()
    
    corpora = args.train
    if (corpora is None):
        parser.error('You need to pass corpora file name to the -train parameter')
    output = args.output
    if (output is None):
        parser.error('You need to pass output file name to the -output parameter')

    
    dm = int(args.cbow)
    size = int(args.size)
    window = int(args.window)
    negative = int(args.negative)
    sample = float(args.sample)
    alpha = float(args.alpha)
    passes = int(args.iter)
    cores = int(args.threads)
    
    min_count = int(args.min_count)

    train_docs, dev_docs, test_docs, alldocs = load_data()

    print('%d docs: %d train, %d test' % (len(alldocs), len(train_docs), len(test_docs)))
    #cores = multiprocessing.cpu_count()
    run_doc2vec(train_docs, dev_docs, test_docs, dm, size, window, alpha, negative, sample, cores, min_count, passes, output)
