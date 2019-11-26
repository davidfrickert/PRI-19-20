import csv
import itertools
import json
import operator
import os
import string
from collections import OrderedDict
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from psutil import cpu_count
from os.path import splitext
from platform import system
from statistics import mean
from xml.dom.minidom import parse

import networkx
import nltk
from networkx import pagerank
from nltk.corpus import stopwords
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

stop_words = set(stopwords.words('english'))

class Helper:
    @staticmethod
    def dictToOrderedList(d: dict, rev=False):
        return sorted(d.items(), key=operator.itemgetter(1), reverse=rev)

    @staticmethod
    def getMaxDiff(pr0, pr1):
        return max(map(float.__sub__, pr1, pr0))

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
            result += ' '.join([t.getElementsByTagName('word')[0].firstChild.nodeValue for t in tokens])
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
    def results(pr, reference, nvals=50):
        avg_prec = {}
        prec = {}
        re = {}
        f1 = {}

        with open(reference) as f:
            reference_results = json.load(f)
            for doc_name in pr.keys():
                doc_results = set(itertools.chain.from_iterable(reference_results[doc_name]))
                kf_results = set([Helper.stemKF(kf) for kf in pr[doc_name][:nvals]])
                y_true = [1 if kf in doc_results else 0 for kf in (kf_results.union(doc_results))]
                y_score = [1 if kf in kf_results else 0 for kf in (kf_results.union(doc_results))]
                if len(y_score) > len(y_true):
                    y_score = y_score[:y_true]
                else:
                    while len(y_score) < len(y_true):
                        y_score.append(0)
                avg_prec.update({doc_name: average_precision_score(y_true, y_score)})
                prec.update({doc_name: precision_score(y_true, y_score)})
                re.update({doc_name: recall_score(y_true, y_score)})
                f1.update({doc_name: f1_score(y_true, y_score)})

        meanAPre = mean(avg_prec.values())
        meanPrec = mean(prec.values())
        meanRe = mean(re.values())
        meanF1 = mean(f1.values())

        return meanAPre, meanPrec, meanRe, meanF1


def buildGramsUpToN(doc, n):
    ngrams = []
    doc = doc.lower()

    s = [sentence.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for sentence in
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


def buildGraph(doc, co_occurence_window=2):
    print('started graph')

    # Step 1
    # Read document and build ngrams. Results is List of List of ngrams (each inner List is all ngrams of sentence)
    ngrams = buildGramsUpToN(doc, 3)
    print('Ngrams', ngrams)

    # Step 2
    # Build Graph and add all ngrams
    # from_iterable transforms [ [ng1, ng2], [ng3, ng4], ...] to [ ng1, ng2, ng3, ng4 ]
    # OrderedDict instead of set() for determinisic behaviour (every execution results in the same order in nodes)
    # deletes duplicates
    g = networkx.Graph()

    g.add_nodes_from(list(OrderedDict.fromkeys((itertools.chain.from_iterable(ngrams)))))

    print('Nodes', g.nodes)

    # Step 3
    # Add edges
    # for each phrase (ngrams[i] is a list of words in phrase 'i') get all combinations of all words as edges
    [g.add_edges_from(itertools.combinations(ngrams[i], 2)) for i in range(len(ngrams))]

    # print('Edges', g.edges)

    pr = getKeyphrasesFromGraph(g, n_iter=1)

    print('Final PR', pr)

    # pos = networkx.spring_layout(g)
    #
    # networkx.draw_networkx_nodes(g, pos, node_size=2000)
    #
    # networkx.draw_networkx_edges(g, pos, edgelist=g.edges,
    #                              width=6)
    #
    # networkx.draw_networkx_labels(g, pos, font_size=15, font_family='sans-serif')
    # plt.axis('off')
    # plt.show()
    return pr


def getKeyphrasesFromGraph(g: networkx.Graph, n_iter=1, d=0.15):
    # N is the total number of candidates

    N = len(g.nodes)
    pr = pagerank(g)

    # cand := pi, calculate PR(pi) for all nodes
    for _ in range(n_iter):
        pr_pi = {}
        for cand in g.nodes:
            pi_links = g.edges(cand)
            # e = (pi, pj)
            # pj = e[1]
            # PR(pj) = pr[pj] = pr[e[1]]
            # Links(pj) = g.edges(pj) = g.edges(e[1])
            sum_pr_pj = sum([pr[e[1]] / len(g.edges(e[1])) for e in pi_links])
            pr_pi[cand] = d / N + (1 - d) * sum_pr_pj

        print(Helper.dictToOrderedList(pr_pi, rev=True))

        pr = pr_pi
    with open('pr.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in pr.items():
            writer.writerow([k, v])
    return list(OrderedDict(Helper.dictToOrderedList(pr, rev=True)).keys())


def main():
    test = Helper.getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test')
    # with open('nyt.txt', 'r') as doc:
    #     doc = ' '.join(doc.readlines())

    # if windows do single process because multiprocess not working
    if system() == 'Windows':
        single_process(test)
    else:
        multi_process(test)


def multi_process(test):
    with ProcessPoolExecutor(max_workers=cpu_count(logical=False)) as executor:
        fts = {}
        kfs = {}
        for file in test:
            fts.update({executor.submit(buildGraph, test[file].lower()): file})
        print('shit')
        for future in as_completed(fts):
            file = fts[future]
            kfs.update({file: future.result()})
    meanAPre, meanPre, meanRe, meanF1 = Helper.results(kfs, 'ake-datasets-master/datasets/500N-KPCrowd/references/test.reader.stem.json')
    print(f'Mean Avg Pre for {len(kfs.keys())} documents: ', meanAPre)
    print(f'Mean Precision for {len(kfs.keys())} documents: ', meanPre)
    print(f'Mean Recall for {len(kfs.keys())} documents: ', meanRe)
    print(f'Mean F1 for {len(kfs.keys())} documents: ', meanF1)

def single_process(test):
    kfs = {}
    for file in test:
        kf = buildGraph(test[file].lower())
        kfs.update({file: kf})

    meanAPre, meanPre, meanRe, meanF1= Helper.results(kfs, 'ake-datasets-master/datasets/500N-KPCrowd/references/test.reader.stem.json')
    print(f'Mean Avg Pre for {len(kfs.keys())} documents: ', meanAPre)
    print(f'Mean Precision for {len(kfs.keys())} documents: ', meanPre)
    print(f'Mean Recall for {len(kfs.keys())} documents: ', meanRe)
    print(f'Mean F1 for {len(kfs.keys())} documents: ', meanF1)



if __name__ == '__main__':
    # single threaded => 142.234375
    import time

    start = time.time()
    main()
    print(time.time() - start)
