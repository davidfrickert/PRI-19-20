import csv
import itertools
import operator
import os
import string
from collections import OrderedDict
from os.path import splitext
from xml.dom.minidom import parse

import matplotlib.pyplot as plt
import networkx
import nltk
from networkx import pagerank
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

vec = None
X = None
terms = None
scoreArr = None


class Helper:
    @staticmethod
    def dictToOrderedList(d: dict, rev=False):
        return sorted(d.items(), key=operator.itemgetter(1), reverse=rev)


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


def buildGraph(filename, co_occurence_window=2):
    with open(filename, 'r') as doc:
        # Step 1
        # Read document and build ngrams. Results is List of List of ngrams (each inner List is all ngrams of sentence)
        doc = ' '.join(doc.readlines())

        ngrams = buildGramsUpToN(doc, 3)

        print('Ngrams', ngrams)

        # Step 2
        # Build Graph and add all ngrams
        # from_iterable transforms [ [ng1, ng2], [ng3, ng4], ...] to [ ng1, ng2, ng3, ng4 ]
        # OrderedDict instead of set() for determinisic behaviour (every execution results in the same order in nodes)
        # deletes duplicates
        g = networkx.Graph()

        g.add_nodes_from(list(OrderedDict.fromkeys((itertools.chain.from_iterable(ngrams)))))

        print('Nodes', g.nodes())

        # Step 3
        # Add edges
        # for each phrase (ngrams[i] is a list of words in phrase 'i') get all combinations of all words as edges
        [g.add_edges_from(itertools.combinations(ngrams[i], 2)) for i in range(len(ngrams))]

        # print('Edges', g.edges)

        pr = undirectedPR(g, n_iter=1)

        print('Final PR', pr)
        #
        # pos = networkx.spring_layout(g)
        #
        # networkx.draw_networkx_nodes(g, pos, node_size=2000)
        #
        # networkx.draw_networkx_edges(g, pos, edgelist=g.edges(),
        #                              width=6)
        #
        # networkx.draw_networkx_labels(g, pos, font_size=15, font_family='sans-serif')
        # plt.axis('off')
        # plt.show()


def buildGraphNew(doc, collection: dict):
    ngrams = buildGramsUpToN(doc, 3)
    doc_i = collection.index(doc)

    print('Ngrams', ngrams)

    # Step 2
    # Build Graph and add all ngrams
    # from_iterable transforms [ [ng1, ng2], [ng3, ng4], ...] to [ ng1, ng2, ng3, ng4 ]
    # OrderedDict instead of set() for determinisic behaviour (every execution results in the same order in nodes)
    # deletes duplicates
    g = networkx.Graph()

    g.add_nodes_from(list(OrderedDict.fromkeys((itertools.chain.from_iterable(ngrams)))))

    print('Nodes', g.nodes())

    # Step 3
    # Add edges
    # for each phrase (ngrams[i] is a list of words in phrase 'i') get all combinations of all words as edges
    edges = list(itertools.chain.from_iterable([itertools.combinations(ngrams[i], 2) for i in range(len(ngrams))]))
    new_edges = addWeights(edges, collection)
    print(new_edges)
    g.add_weighted_edges_from(new_edges)

    # print('Edges', g.edges)

    pr = undirectedPR(g, doc_i, n_iter=1)

    print('Final PR', pr)

    pos = networkx.spring_layout(g)

    networkx.draw_networkx_nodes(g, pos, node_size=2000)

    networkx.draw_networkx_edges(g, pos, edgelist=g.edges(),
                                 width=6)

    networkx.draw_networkx_labels(g, pos, font_size=15, font_family='sans-serif')
    plt.axis('off')
    plt.show()


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


def convertXMLToTaggedSents(xml):
    result = []

    for i, sentence in enumerate(xml):
        tokens = sentence.getElementsByTagName('token')
        result.append([(t.getElementsByTagName('word')[0].firstChild.nodeValue,
                        t.getElementsByTagName('POS')[0].firstChild.nodeValue) for t in tokens])
    return result


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
                docs.update({doc_name: convertXML(xml)})
            else:
                docs.update({doc_name: convertXMLToTaggedSents(xml)})

    return docs


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


def addWeights(edges, collection):
    weights = co_ocurrence(edges, collection)
    n_edges = [e + (w,) for (e, w) in zip(edges, weights)]
    return n_edges


def co_ocurrence(edges, collection):

    co_oc = []
    for doc in collection:
        sentences = nltk.sent_tokenize(doc)
        for sent in sentences:
            for i, edge in enumerate(edges):
                not_in_vec = len(co_oc) < i + 1
                w0, w1 = edge
                co_oc_n = 0 if not_in_vec else co_oc[i]
                if w0 in sent and w1 in sent:
                    co_oc_n += 1
                if not_in_vec:
                    co_oc.append(co_oc_n)
                else:
                    co_oc[i] = co_oc_n
    return co_oc


def undirectedPR(g: networkx.Graph, doc_i: int, n_iter=1, d=0.15,):
    # N is the total number of candidates
    N = len(g.nodes())
    pr = pagerank(g)

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

            div = sum([prior(doc_i, w) for w in pi_links])
            if div == 0:
                div = 1

            pr_pi[pi] = d * (prior(doc_i, pi) / div) + (1 - d) * rst

            # pr_pi[cand] = d / N + (1 - d) * sum_pr_pj

        print(Helper.dictToOrderedList(pr_pi, rev=True))

        pr = pr_pi
    with open('pr.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in pr.items():
            writer.writerow([k, v])
    return OrderedDict(Helper.dictToOrderedList(pr_pi, rev=True)[:5])


def main():
    global X, vec, terms, scoreArr

    test = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test')
    print(test.values())
    vec = TfidfVectorizer(stop_words=None, ngram_range=(1, 3))
    X = vec.fit_transform(test.values())
    terms = vec.get_feature_names()
    scoreArr = X.toarray()

    for file in test:
        buildGraphNew(test[file].lower(), list(map(lambda doc: doc.lower(), test.values())))
        break


main()
