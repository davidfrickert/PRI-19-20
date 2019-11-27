import csv
import itertools
import string
from collections import OrderedDict
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from platform import system
from time import time

import networkx
import nltk
from networkx import pagerank
from nltk.corpus import stopwords
from psutil import cpu_count
from sklearn.feature_extraction.text import TfidfVectorizer

from project_p2_ex1 import Helper

vec = None
X = None
terms = None
scoreArr = None
stop_words = set(stopwords.words('english'))


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


def buildGraphNew(doc, collection: list):
    ngrams = buildGramsUpToN(doc, 3)
    doc_i = collection.index(doc)

    # print('Ngrams', ngrams)

    # Step 2
    # Build Graph and add all ngrams
    # from_iterable transforms [ [ng1, ng2], [ng3, ng4], ...] to [ ng1, ng2, ng3, ng4 ]
    # OrderedDict instead of set() for determinisic behaviour (every execution results in the same order in nodes)
    # deletes duplicates
    g = networkx.Graph()

    g.add_nodes_from(list(OrderedDict.fromkeys((itertools.chain.from_iterable(ngrams)))))

    # print('Nodes', g.nodes())

    # Step 3
    # Add edges
    # for each phrase (ngrams[i] is a list of words in phrase 'i') get all combinations of all words as edges
    edges = list(itertools.chain.from_iterable([itertools.combinations(ngrams[i], 2) for i in range(len(ngrams))]))
    new_edges = addWeights(edges, collection)
    g.add_weighted_edges_from(new_edges)

    # print('Edges', g.edges)

    return g


# encontra o termo e retorna o seu score idf
def prior(index, p):
    global vec, X, terms, scoreArr

    for i, term in enumerate(terms):
        if term == p:
            return scoreArr[index][i]

    return 0


def priorCandLocation(cand: str, doc: str):
    # normalized location, inverted because sentences in first keyphrases
    # are more relevant
    nsw_doc = Helper.filterStopWords(doc)
    first_match = 1 - (nsw_doc.find(cand) / len(nsw_doc))
    if first_match == 0:
        print('bad')
    return first_match * len(cand)


def addWeights(edges, collection):
    weights = co_ocurrence(edges, collection)
    n_edges = [e + (w,) for (e, w) in zip(edges, weights)]
    return n_edges


def co_ocurrence(edges, collection):
    co_oc = []
    for doc in collection:
        sentences = nltk.sent_tokenize(doc)
        sentences = [sentence.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                     for sentence in sentences]
        new_sents = []
        for sent in sentences:
            new_sents.append(' '.join([word for word in nltk.word_tokenize(sent) if word not in stop_words]))
        sentences = new_sents

        for sent in sentences:
            for i, edge in enumerate(edges):
                not_in_vec = len(co_oc) < i + 1
                w0, w1 = edge
                co_oc_n = 0 if not_in_vec else co_oc[i]
                # if w0 == 'york 3':
                #     print('h')
                if w0 in sent and w1 in sent:
                    co_oc_n += 1
                if not_in_vec:
                    co_oc.append(co_oc_n)
                else:
                    co_oc[i] = co_oc_n
    return co_oc


def getKeyphrases(doc: str, collection: list):

    doc_i = collection.index(doc)
    print(f'started graph for doc nº {doc_i}')
    g = buildGraphNew(doc, collection)
    # change n_iter, 1 for now for testing purposes
    kf = getKeyphrasesFromGraph(g, doc, doc_i, n_iter=1)
    print(f'finished graph for doc nº {doc_i}')
    return kf


def getKeyphrasesFromGraph(g: networkx.Graph, doc: str, doc_i: int, n_iter=1, d=0.15):
    # N is the total number of candidates
    N = len(g.nodes())
    pr = pagerank(g)

    diff_list = []

    # cand := pi, calculate PR(pi) for all nodes
    for _ in range(n_iter):
        pr_pi = {}
        for pi in g.nodes():
            pi_links = list(map(lambda e: e[1], g.edges(pi)))

            # e = (pi, pj)
            # pj = e[1]
            # PR(pj) = pr[pj] = pr[e[1]]
            # Links(pj) = g.edges(pj) = g.edges(e[1])

            # sum_pr_pj = sum([pr[e[1]] / len(g.edges(e[1])) for e in pi_links])
            # todos os candidatos ou so os que estao ligados ao cand agora
            # ∑pjPrior(pj)
            rst_array = []
            for pj in pi_links:
                pj_links = list(map(lambda e: e[1], g.edges(pj)))
                bot = sum([g.edges[pj, pk]['weight'] for pk in pj_links])
                if bot == 0:
                    continue
                top = pr[pj] * g.edges[pj, pi]['weight']
                rst = top / bot
                rst_array.append(rst)
            rst = sum(rst_array)

            # div = sum([prior(doc_i, w) for w in pi_links])
            div = sum([priorCandLocation(w, doc) for w in pi_links])

            if div == 0:
                div = 1

            # pr_pi[pi] = d * (prior(doc_i, pi) / div) + (1 - d) * rst
            pr_pi[pi] = d * (priorCandLocation(pi, doc) / div) + (1 - d) * rst
            # pr_pi[cand] = d / N + (1 - d) * sum_pr_pj

        diff = Helper.getMaxDiff(pr.values(), pr_pi.values())
        sum_pr = sum(pr_pi.values())

        print(Helper.dictToOrderedList(pr_pi, rev=True))
        print('Max diff', diff)
        print('sum of PR', sum_pr)

        # pr = old iter, pr_pi = new iter
        diff_list.append(diff)
        pr = pr_pi
        # converges at max rate of 0.01
        if diff < 0.01:
            break
    with open('pr.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in pr.items():
            writer.writerow([k, v])
    return list(OrderedDict(Helper.dictToOrderedList(pr, rev=True)).keys())


def single_process(ds):
    kfs = {}
    for file in ds:
        d_pr = getKeyphrases(ds[file].lower(), list(map(lambda doc: doc.lower(), ds.values())))
        kfs.update({file: d_pr})

    meanAPre, meanPre, meanRe, meanF1 = Helper.results(kfs, 'ake-datasets-master/datasets/500N-KPCrowd/references'
                                                            '/test.reader.stem.json')
    print(f'Mean Avg Pre for {len(kfs.keys())} documents: ', meanAPre)
    print(f'Mean Precision for {len(kfs.keys())} documents: ', meanPre)
    print(f'Mean Recall for {len(kfs.keys())} documents: ', meanRe)
    print(f'Mean F1 for {len(kfs.keys())} documents: ', meanF1)


def multi_process(ds):
    with ProcessPoolExecutor(max_workers=cpu_count(logical=Helper.logical())) as executor:
        fts = {}
        kfs = {}
        for file in ds:
            fts.update({executor.submit(getKeyphrases, ds[file].lower(),
                                        list(map(lambda doc: doc.lower(), ds.values()))): file})
        for future in as_completed(fts):
            file = fts[future]
            kfs.update({file: future.result()})
    meanAPre, meanPre, meanRe, meanF1 = Helper.results(kfs,
                                                       'ake-datasets-master/datasets/500N-KPCrowd/references/test'
                                                       '.reader.stem.json')
    print(f'Mean Avg Pre for {len(kfs.keys())} documents: ', meanAPre)
    print(f'Mean Precision for {len(kfs.keys())} documents: ', meanPre)
    print(f'Mean Recall for {len(kfs.keys())} documents: ', meanRe)
    print(f'Mean F1 for {len(kfs.keys())} documents: ', meanF1)


def main():
    global X, vec, terms, scoreArr

    test = Helper.getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test')

    vec = TfidfVectorizer(stop_words=None, ngram_range=(1, 3))
    X = vec.fit_transform(test.values())
    terms = vec.get_feature_names()
    scoreArr = X.toarray()

    # if windows and not do logical (checks v >= 3.8.0)
    if system() == 'Windows' and not Helper.logical():
        single_process(test)
    # if linux/mac or windows with v >= 3.8.0
    else:
        multi_process(test)


if __name__ == '__main__':
    start = time()
    main()
    print(time() - start)
