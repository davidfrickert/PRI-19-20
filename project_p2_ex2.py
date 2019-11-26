import csv
import itertools
import json
import string
from collections import OrderedDict
from statistics import mean

import networkx
import nltk
from networkx import pagerank
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score

from project_p2_ex1 import Helper

vec = None
X = None
terms = None
scoreArr = None


def buildGramsUpToN(doc, n):
    ngrams = []
    doc = doc.lower()
    invalid = set(stopwords.words('english'))

    s = [sentence.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for sentence in
         nltk.sent_tokenize(doc)]
    sents = [nltk.word_tokenize(_) for _ in s]

    # remove stop_words

    for sentence in sents:
        tormv = []
        for word in sentence:
            if word in invalid:
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
    print(f'started graph for doc nº {doc_i}')
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

    pr = undirectedPR(g, doc, doc_i, n_iter=1)

    # print('Final PR', pr)

    return pr


# encontra o termo e retorna o seu score idf
def prior(index, p):
    global vec, X, terms, scoreArr
    # stopW = set(stopwords.words('english'))

    # return merge(dataset, terms, scoreArr)

    # print(terms)
    # print(scoreArr)

    for i, term in enumerate(terms):
        if term == p:
            # print(term)
            # print(p)
            # print(scoreArr[0][i])
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
    stop = set(stopwords.words('english'))
    co_oc = []
    for doc in collection:
        sentences = nltk.sent_tokenize(doc)
        sentences = [sentence.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                     for sentence in sentences]
        new_sents = []
        for sent in sentences:
            new_sents.append(' '.join([word for word in nltk.word_tokenize(sent) if word not in stop]))
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


def undirectedPR(g: networkx.Graph, doc: str, doc_i: int, n_iter=1, d=0.15):
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
    return list(OrderedDict(Helper.dictToOrderedList(pr_pi, rev=True)).keys())


def main():
    global X, vec, terms, scoreArr

    test = Helper.getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test')

    vec = TfidfVectorizer(stop_words=None, ngram_range=(1, 3))
    X = vec.fit_transform(test.values())
    terms = vec.get_feature_names()
    scoreArr = X.toarray()

    pr = {}
    threads = []
    # MAX_THREADS = 8
    # with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    for file in test:
        # future = executor.submit(buildGraphNew, test[file].lower(), list(map(lambda doc: doc.lower(), test.values())))
        # threads.append(future)
        d_pr = buildGraphNew(test[file].lower(), list(map(lambda doc: doc.lower(), test.values())))
        pr.update({file: d_pr})
        break

    meanAPre = Helper.results(pr, 'ake-datasets-master/datasets/500N-KPCrowd/references/test.reader.stem.json')
    print(f'Mean Avg Pre for {len(pr.keys())} documents: ', meanAPre)
    # for file, future in zip(test.keys(), threads):
    #    pr.update({file: future.result()})


main()
