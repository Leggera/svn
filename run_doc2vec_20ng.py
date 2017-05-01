import gensim
from collections import namedtuple
from run_doc2vec import run_doc2vec
import argparse
from sklearn.datasets import fetch_20newsgroups

def normalize_text(text):
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text

def get_data(subset):
    SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
    newsgroups_data = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    docs = []
    for news_no, news in enumerate(newsgroups_data.data):    
        tokens = gensim.utils.to_unicode(normalize_text(news)).split()
        
        if len(tokens) == 0:
            continue
        
        split = subset
        sentiment =  newsgroups_data.target[news_no]
        tags = [subset + ' ' + 'SENT_'+ str(news_no) + " " + str(sentiment)]

        docs.append(SentimentDocument(tokens, tags, split, sentiment))
    return docs

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
    
    args = parser.parse_args()

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
    

    train_docs = get_data('train')
    test_docs = get_data('test')

    alldocs = train_docs + test_docs

    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))

    run_doc2vec(train_docs, test_docs, alldocs, dm, size, window, alpha, negative, sample, cores, min_count, passes, output)

