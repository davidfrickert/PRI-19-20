import itertools
import pprint
import sys
import time
from math import log

import nltk
import string

import numpy

from sklearn.feature_extraction.text import TfidfVectorizer, _document_frequency

from project_ex2 import getDataFromDir, calcMetrics, merge, mergeDict

numpy.set_printoptions(threshold=sys.maxsize)


def findBiggestGram(candidates):
    max_gram = 0
    for doc in candidates:
        for cand in doc:
            gram = len(cand.split())
            max_gram = gram if gram > max_gram else max_gram
    return max_gram


def getAllChunks(tagged_sents, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}', deliver_as='list'):
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    chunker = nltk.chunk.regexp.RegexpParser(grammar)

    all_chunks = [nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                  for tagged_sent in tagged_sents]

    if deliver_as == 'sentences':
        candidates = []
        for sentence in all_chunks:
            candidates.append(set([' '.join(word for word, pos, chunk in group).lower()
                                           for key, group in itertools.groupby(sentence, lambda tpl: tpl[2] != 'O') if
                                           key]))
        return [[cand for cand in sent if cand not in stop_words and not all(char in punct for char in cand)]
                for sent in candidates]
              #  ]
    else:
        candidates = set()
        for sentence in all_chunks:
            candidates = candidates | set([' '.join(word for word, pos, chunk in group).lower()
                                           for key, group in itertools.groupby(sentence, lambda tpl: tpl[2] != 'O') if
                                           key])

        return [cand for cand in candidates
                if cand not in stop_words and not all(char in punct for char in cand)]

    # all_chunks is a list of lists. the inner lists are chunks for each sentence (so we don't have multi-sentence candidates)



def getTFIDFScore(dataset, mergetype='list'):
    # extract candidates from each text in texts, either chunks or words

    ds = list(itertools.chain.from_iterable(getAllCandidates(dataset, deliver_as='sentences')))
    words = listOfTaggedToListOfWords(dataset)

    vec = TfidfVectorizer(tokenizer=lambda i: i, lowercase=False)

    vec.fit(ds)

    vec.ngram_range = (1, findBiggestGram(ds))

    X = vec.transform(words).toarray()

    # print(X)

    terms = vec.get_feature_names()

    if mergetype == 'dict':
        return mergeDict(dataset, terms, X)
    else:
        return merge(dataset, terms, X)


def getAllCandidates(dataset, deliver_as='list'):
    return [getAllChunks(text, deliver_as=deliver_as) for text in dataset.values()]


def listOfTaggedToString(dataset):
    documents = []

    for d in dataset.values():
        arr = []
        for p in d:
            arr.append([k[0].lower() for k in p])
        documents.append(' '.join(itertools.chain.from_iterable(arr)))
    return documents


def listOfTaggedToListOfWords(dataset):
    documents = []
    punct = set(string.punctuation)
    for d in dataset.values():
        doc_i = []
        for ph in d:
            for w_tag in ph:
                word = w_tag[0].lower()
                if not all(char in punct for char in word):
                    doc_i.append(word)
        documents.append(doc_i)
    return documents


def getBM25Score(dataset, k1=1.2, b=0.75, mergetype='list', min_df=2, cands=None):
    if not cands:
        cands = getAllCandidates(dataset, deliver_as='sentences')
        ds = [list(itertools.chain.from_iterable(doc)) for doc in cands]
    else:
        ds = cands
    words = listOfTaggedToListOfWords(dataset)
    # documents = listOfTaggedToString(dataset)
    # stopW = set(nltk.corpus.stopwords.words('english'))

    vec_tf = TfidfVectorizer(tokenizer=lambda e: e, lowercase=False, use_idf=False)

    vec_tf.fit(ds)

    #
    vec_tf.ngram_range = (1, findBiggestGram(ds))
    # vec_tf.tokenizer = None
    # vec_tf.stop_words = stopW
    # vec_tf.min_df = 2

    terms = vec_tf.get_feature_names()

    X = vec_tf.transform(words)

    tf_arr = X.toarray()

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
            if DF >= min_df:
                temp.append(bm25 * (len(terms[j]) / len(terms[j].split())))
            else:
                temp.append(0.)
        score.append(temp)

    if mergetype == 'dict':
        return mergeDict(dataset, terms, score)
    else:
        return merge(dataset, terms, score)


def getAvgDL(all_d):
    return numpy.average([len(d) for d in all_d])


def main():
    train = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/train', mode='list')
    results = getBM25Score(train)
    results_ = getTFIDFScore(train)

    calcMetrics(results, 'ake-datasets-master/datasets/500N-KPCrowd/references/train.reader.stem.json')
    calcMetrics(results_, 'ake-datasets-master/datasets/500N-KPCrowd/references/train.reader.stem.json')


if __name__ == '__main__':
    main()
