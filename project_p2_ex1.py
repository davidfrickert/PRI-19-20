import itertools
import itertools
import json
import operator
import os
import string
from collections import OrderedDict
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from os.path import splitext
from platform import system
from statistics import mean
from sys import version_info
from xml.dom.minidom import parse

import networkx
import nltk
from nltk.corpus import stopwords
from psutil import cpu_count
from sklearn.metrics import precision_score, recall_score, f1_score

stop_words = set(stopwords.words('english'))


class Helper:
    @staticmethod
    def dictToOrderedList(d: dict, rev=False):
        return sorted(d.items(), key=operator.itemgetter(1), reverse=rev)

    @staticmethod
    def checkConvergence(pr0, pr1, N, rate):
        return sum(map(abs, map(float.__sub__, pr1, pr0))) < N * rate

    @staticmethod
    def stemKF(kf):
        stemmer = nltk.PorterStemmer()
        words = nltk.word_tokenize(kf)
        return ' '.join([stemmer.stem(w) for w in words])

    @staticmethod
    def filterStopWords(doc: str):
        return ' '.join([word for word in nltk.word_tokenize(doc) if word not in stop_words])

    @staticmethod
    def getDataFromDir(path, mode='string'):
        directory = os.fsencode(path)

        docs = {}

        for f in os.listdir(directory):
            filePath = path + '/' + f.decode("utf-8")

            with open(filePath, encoding='utf8') as datasource:
                dom = parse(datasource)
                xml = dom.getElementsByTagName('sentence')

                doc_name = splitext(f)[0].decode("utf-8")
                if mode == 'string':
                    docs.update({doc_name: Helper.convertXML(xml)})
                else:
                    docs.update({doc_name: Helper.convertXMLToTaggedSents(xml)})

        return docs

    @staticmethod
    def convertXML(xml):
        result = ""

        for i, sentence in enumerate(xml):
            SEPARATOR = '\r\n'
            tokens = sentence.getElementsByTagName('token')
            result += ' '.join([t.getElementsByTagName('lemma')[0].firstChild.nodeValue for t in tokens])
            if not result.endswith('.'):
                result += '.'
            result += SEPARATOR

        return result

    @staticmethod
    def convertXMLToTaggedSents(xml):
        result = []

        for i, sentence in enumerate(xml):
            tokens = sentence.getElementsByTagName('token')
            result.append([(t.getElementsByTagName('lemma')[0].firstChild.nodeValue,
                            t.getElementsByTagName('POS')[0].firstChild.nodeValue) for t in tokens])
        return result

    @staticmethod
    def getTrueKeyphrases(reference):
        with open(reference) as f:
            reference_results = json.load(f)
            return {k: list(itertools.chain.from_iterable(v)) for k, v in reference_results.items()}

    @staticmethod
    def results(res, reference):
        avg_prec = {}
        prec = {}
        re = {}
        f1 = {}
        avg = {}
        with open(reference) as f:
            reference_results = json.load(f)
            for doc_name in res.keys():
                pr = res[doc_name]
                # nvals = int(0.1 * len(g.nodes))
                nvals = 50
                doc_results = set(itertools.chain.from_iterable(reference_results[doc_name]))
                print(nvals - len(doc_results))
                # print(nvals / len(doc_results))
                kf_results = set([Helper.stemKF(kf) for kf in pr[:nvals]])

                y_true = [1 if kf in doc_results else 0 for kf in (kf_results.union(doc_results))]
                y_score = [1 if kf in kf_results else 0 for kf in (kf_results.union(doc_results))]

                avg_prec.update({doc_name: Metrics.averagePrecision(list(kf_results), list(doc_results))})
                # avg_prec.update({doc_name: average_precision_score(y_true, y_score)})
                prec.update({doc_name: precision_score(y_true, y_score)})
                re.update({doc_name: recall_score(y_true, y_score)})
                f1.update({doc_name: f1_score(y_true, y_score)})
                avg[doc_name] = nvals / len(doc_results)

        meanAPre = mean(avg_prec.values())
        meanPrec = mean(prec.values())
        meanRe = mean(re.values())
        meanF1 = mean(f1.values())
        print(mean(avg.values()))

        return meanAPre, meanPrec, meanRe, meanF1

    @staticmethod
    def logical():
        return version_info >= (3, 7, 0)


class Metrics:
    '''
    Class for metrics, mainly for calculating average precision

    @:arg predicted: predicted keyphrases by our programs
        @:type list[str]
    @:arg true: true keyphrases in the references
        @:type list[str]
    '''

    @staticmethod
    def averagePrecision(predicted, true):
        return sum(
            [
                Metrics.precisionAt(predicted, true, k + 1) * Metrics.isKeyphrase(kf, true)
                for k, kf in enumerate(predicted)
            ]
        ) / len(true)

    @staticmethod
    def isKeyphrase(item, true):
        return 1 if item in true else 0

    @staticmethod
    def precisionAt(predicted, true, k):
        true_positives = 0

        tmp = predicted[:k]

        for kf in tmp:
            if kf in true:
                true_positives += 1

        return float(true_positives) / float(k)


def buildGramsUpToN(doc, n):
    ngrams = []
    doc = doc.lower()

    s = [sentence.translate(str.maketrans("", "", string.punctuation + string.digits)) for sentence in
         nltk.sent_tokenize(doc)]

    sents = [nltk.word_tokenize(_) for _ in s]

    # remove stop_words

    for sentence in sents:
        tormv = []
        for word in sentence:
            if word in stop_words:
                tormv.append(word)
        [sentence.remove(w) for w in tormv]

    # end remove stop_words

    for sent in sents:
        ngram_list = []
        for N in range(1, n + 1):
            ngram_list += [sent[i:i + N] for i in range(len(sent) - N + 1)]
        ngrams.append([' '.join(gram) for gram in ngram_list])

    return ngrams


def buildGraph(doc):
    print('started graph')

    # Step 1
    # Read document and build ngrams. Results is List of List of ngrams (each inner List is all ngrams of sentence)
    ngrams = buildGramsUpToN(doc, 3)
    # print('Ngrams', ngrams)

    # Step 2
    # Build Graph and add all ngrams
    # from_iterable transforms [ [ng1, ng2], [ng3, ng4], ...] to [ ng1, ng2, ng3, ng4 ]
    # OrderedDict instead of set() for determinisic behaviour (every execution results in the same order in nodes)
    # deletes duplicates
    g = networkx.Graph()

    g.add_nodes_from(list(OrderedDict.fromkeys((itertools.chain.from_iterable(ngrams)))))

    # print('Nodes', g.nodes)

    # Step 3
    # Add edges
    # for each phrase (ngrams[i] is a list of words in phrase 'i') get all combinations of all words as edges
    # combinationsN(ngrams[0], 2)
    [g.add_edges_from(itertools.combinations(ngrams[i], 2)) for i in range(len(ngrams))]
    # [g.add_edges_from(combinationsN(ngrams[i], 10)) for i in range(len(ngrams))]
    # print('Edges', g.edges)
    return g
    # pr = getKeyphrasesFromGraph(g, n_iter=15)

    # print('Final PR', pr)

    # return pr, g


def combinationsN(list, n):
    combs = []
    for i, word in enumerate(list):
        min_ind = max(0, i - n)
        max_ind = min(len(list), i + n + 1)
        c = list[min_ind:max_ind]
        c.remove(word)
        combs += [(word, w2) for w2 in c]
    return combs


def getPageRankFromGraph(g: networkx.Graph, n_iter=1, d=0.15):
    # N is the total number of candidates

    N = len(g.nodes)
    pr = dict(zip(g.nodes, [1 / len(g.nodes) for _ in range(len(g.nodes))]))
    rate = 0.00001
    # cand := pi, calculate PR(pi) for all nodes
    for _ in range(n_iter):
        pr_pi = {}
        for cand in g.nodes:
            pi_links = g.edges(cand)
            sum_pr_pj = sum([pr[e[1]] / len(g.edges(e[1])) for e in pi_links])
            pr_pi[cand] = d / N + (1 - d) * sum_pr_pj

        isConverged = Helper.checkConvergence(pr.values(), pr_pi.values(), N, rate)
        pr = pr_pi
        if isConverged:
            break
    print(sum(pr.values()))
    return pr


def getKeyphrases(doc):
    g = buildGraph(doc)
    pr = getPageRankFromGraph(g, n_iter=15)
    return list(OrderedDict(Helper.dictToOrderedList(pr, rev=True)).keys())


def multi_process(test):
    with ProcessPoolExecutor(max_workers=cpu_count(logical=Helper.logical())) as executor:
        fts = {}
        kfs = {}
        for file in test:
            fts.update({executor.submit(getKeyphrases, test[file].lower()): file})
        for future in as_completed(fts):
            file = fts[future]
            kfs.update({file: future.result()})
    meanAPre, meanPre, meanRe, meanF1 = Helper.results(kfs,
                                                       'ake-datasets-master/datasets/500N-KPCrowd/references/train.reader.stem.json')
    print(f'Mean Avg Pre for {len(kfs.keys())} documents: ', meanAPre)
    print(f'Mean Precision for {len(kfs.keys())} documents: ', meanPre)
    print(f'Mean Recall for {len(kfs.keys())} documents: ', meanRe)
    print(f'Mean F1 for {len(kfs.keys())} documents: ', meanF1)

# NOT UPDATED, USE MULTI-PROCESS
def single_process(test):
    kfs = {}
    for file in test:
        kf = buildGraph(test[file].lower())
        kfs.update({file: kf})

    meanAPre, meanPre, meanRe, meanF1 = Helper.results(kfs,
                                                       'ake-datasets-master/datasets/500N-KPCrowd/references/train'
                                                       '.reader.stem.json')
    print(f'Mean Avg Pre for {len(kfs.keys())} documents: ', meanAPre)
    print(f'Mean Precision for {len(kfs.keys())} documents: ', meanPre)
    print(f'Mean Recall for {len(kfs.keys())} documents: ', meanRe)
    print(f'Mean F1 for {len(kfs.keys())} documents: ', meanF1)


def main():
    test = Helper.getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/train')
    # with open('nyt.txt', 'r') as doc:
    #     doc = ' '.join(doc.readlines())
    #     test = {'nyt.txt': doc}
    #     print(getKeyphrases(doc)[0][:5])
    # if windows do single process because multiprocess not working

    if system() == 'Windows' and not Helper.logical():
        single_process(test)
    else:
        multi_process(test)


if __name__ == '__main__':
    # single threaded => 142.234375
    import time

    start = time.time()
    main()
    print(time.time() - start)
