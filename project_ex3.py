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

    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))

    candidates = set([' '.join(word for word, pos, chunk in group).lower()
                      for key, group in itertools.groupby(all_chunks, lambda tpl: tpl[2] != 'O') if key])

    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]


def getTFIDFScore(dataset):
    # extract candidates from each text in texts, either chunks or words

    ds = [getAllChunks(text) for text in dataset.values()]

    vec = TfidfVectorizer(tokenizer=lambda i: i, lowercase=False)

    X = vec.fit_transform(ds).toarray()

    terms = vec.get_feature_names()

    return merge(dataset, terms, X)


def getBM25Score(dataset, k1=1.2, b=0.75):
    ds = [getAllChunks(text) for text in dataset.values()]
    # docs = [for d in dataset.values() for p in d for k in p]

    stopW = set(nltk.corpus.stopwords.words('english'))
    documents = []

    for d in dataset.values():
        arr = []
        for p in d:
            arr.append([k[0].lower() for k in p])
        documents.append(' '.join(itertools.chain.from_iterable(arr)))

    vec_tf = TfidfVectorizer(tokenizer=lambda e: e, lowercase=False, use_idf=False)

    vec_tf.fit(ds)

    vec_tf.ngram_range = (1, 5)
    vec_tf.tokenizer = None
    vec_tf.stop_words = stopW
    X = vec_tf.transform(documents)

    tf_arr = X.toarray()
    terms = vec_tf.get_feature_names()

    N = len(dataset)
    avgDL = getAvgDL(ds)
    DF_all = _document_frequency(X)  # .sum()
    score = []

    for i, doc in enumerate(dataset):
        temp = []
        dl = len(list(itertools.chain.from_iterable(doc)))
        for j in range(len(X.toarray()[0])):

            DF = DF_all[j]
            tf = tf_arr[i][j]

            bm25_idf = log((N - DF + 0.5) / (DF + 0.5), 10)
            bm25_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * (dl / avgDL))))
            temp.append(bm25_tf * bm25_idf)
        score.append(temp)
    #  tf = (X[i) / ()
    data = merge(dataset, terms, score)
    pprint.pprint(data['politics_world-20786243'])
    return data

def getAvgDL(all_d):
    return numpy.average([len(d) for d in all_d])

# def getTF(term, doc):


def main():
    test = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test', mode='list')

    results = getBM25Score(test)

    # results = getTFIDFScore(test)
    #
    # print(results['politics_us-20782177'])
    #
    calcMetrics(results, 'ake-datasets-master/datasets/500N-KPCrowd/references/test.reader.json')
    #


if __name__ == '__main__':
    main()
