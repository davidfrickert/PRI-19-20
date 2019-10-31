import itertools
import pprint
import sys
from math import log

import gensim
import nltk
import string

import numpy

from sklearn.feature_extraction.text import TfidfVectorizer, _document_frequency

from project_ex2 import getDataFromDir, calcMetrics, merge

numpy.set_printoptions(threshold=sys.maxsize)


def getAllChunks(tagged_sents, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    chunker = nltk.chunk.regexp.RegexpParser(grammar)

    all_chunks = [nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents]
    candidates = set()

    # all_chunks is a list of lists. the inner lists are chunks for each sentence (so we don't have multi-sentence candidates)
    for sentence in all_chunks:
        candidates = candidates | set([' '.join(word for word, pos, chunk in group).lower()
                        for key, group in itertools.groupby(sentence, lambda tpl: tpl[2] != 'O') if key])

    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]


def getAllCandidates(dataset):
    return [getAllChunks(text) for text in dataset.values()]
    

def getBM25Score(dataset, k1=1.2, b=0.75):
    ds = getAllCandidates(dataset)

    stopW = set(nltk.corpus.stopwords.words('english'))
    documents = []

    for d in dataset.values():
        arr = []
        for p in d:
            arr.append([k[0].lower() for k in p])
        documents.append(' '.join(itertools.chain.from_iterable(arr)))

    vec_tf = TfidfVectorizer(tokenizer=lambda e: e, lowercase=False, use_idf=False)

    vec_tf.fit(ds)

    vec_tf.ngram_range = (1, 2)
    vec_tf.tokenizer = None
    vec_tf.stop_words = stopW
    vec_tf.min_df = 2
    X = vec_tf.transform(documents)

    tf_arr = X.toarray()
    terms = vec_tf.get_feature_names()

    N = len(dataset)
    avgDL = getAvgDL(ds)
    DF_all = _document_frequency(X)  # .sum()
    score = []

    for i, doc in enumerate(dataset.values()):
        temp = []
        dl = len(list(itertools.chain.from_iterable(doc)))

        for j in range(len(terms)):
            DF = DF_all[j]
            tf = tf_arr[i][j]

            bm25_idf = log((N - DF + 0.5) / (DF + 0.5), 10)
            bm25_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * (dl / avgDL))))

            bm25 = bm25_tf * (bm25_idf + 1.) 
            temp.append(bm25 * (len(terms[j]) / len(terms[j].split())))
        score.append(temp)
    data = merge(dataset, terms, score)

    return data


def getAvgDL(all_d):
    return numpy.average([len(d) for d in all_d])


def main():
    test = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test', mode='list')
    results = getBM25Score(test)

    calcMetrics(results, 'ake-datasets-master/datasets/500N-KPCrowd/references/test.reader.stem.json')
    

if __name__ == '__main__':
    main()
