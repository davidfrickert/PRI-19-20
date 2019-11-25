import json
import operator
from collections import OrderedDict
import csv

import os
from os.path import  splitext
from xml.dom.minidom import parse, parseString
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx
from networkx import pagerank
import nltk
import string
from nltk.corpus import stopwords
import itertools

import matplotlib.pyplot as plt

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

        pos = networkx.spring_layout(g)

        networkx.draw_networkx_nodes(g, pos, node_size=2000)

        networkx.draw_networkx_edges(g, pos, edgelist=g.edges(),
                                     width=6)

        networkx.draw_networkx_labels(g, pos, font_size=15, font_family='sans-serif')
        plt.axis('off')
        plt.show()


def buildGraphNew(doc, co_occurence_window=2):
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

    pr = undirectedPR(g, n_iter=1, doc = doc)

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
        tokens = sentence.getElementsByTagName('token')
        result += ' '.join([t.getElementsByTagName('word')[0].firstChild.nodeValue for t in tokens])

    return result

def convertXMLToTaggedSents(xml):
    result = []

    for i, sentence in enumerate(xml):
        tokens = sentence.getElementsByTagName('token')
        result.append([(t.getElementsByTagName('word')[0].firstChild.nodeValue,
                        t.getElementsByTagName('POS')[0].firstChild.nodeValue) for t in tokens ])
    return result

def getDataFromDir(path, mode='string'):
    directory = os.fsencode(path)

    docs = {}

    for f in os.listdir(directory):
        filePath = path + '/' + f.decode("utf-8")

        with open(filePath) as datasource:
            dom = parse(datasource)
            xml = dom.getElementsByTagName('sentence')

            doc_name = splitext(f)[0].decode("utf-8")
            if mode == 'string':
                docs.update({doc_name: convertXML(xml)})
            else:
                docs.update({doc_name: convertXMLToTaggedSents(xml)})

    return docs

#encontra o termo e retorna o seu score idf
def prior(doc, p):
    global vec,X,terms,scoreArr
    #stopW = set(stopwords.words('english'))

    
    #return merge(dataset, terms, scoreArr)

    #print(terms)
    #print(scoreArr)

    for i,term in enumerate(terms):
        if term == p:
            #print(term)
            #print(p)
            #print(scoreArr[0][i])
            return scoreArr[0][i]

    return 0

    

def undirectedPR(g: networkx.Graph, n_iter=1, d=0.15, doc=None):
    # N is the total number of candidates
    N = len(g.nodes())
    pr = pagerank(g)

    # cand := pi, calculate PR(pi) for all nodes
    for _ in range(n_iter):
        pr_pi = {}
        for cand in g.nodes():
            pi_links = g.edges(cand)
            # e = (pi, pj)
            # pj = e[1]
            # PR(pj) = pr[pj] = pr[e[1]]
            # Links(pj) = g.edges(pj) = g.edges(e[1])
            
            #sum_pr_pj = sum([pr[e[1]] / len(g.edges(e[1])) for e in pi_links])
            #todos os candidatos ou so os que estao ligados ao cand agora
            #∑pjPrior(pj)
            div = sum([prior(doc, e) for e in pi_links])
            if div == 0:
                div = 1

            pr_pi[cand] = d * (prior(doc, cand) / div) + (1-d)

            #pr_pi[cand] = d / N + (1 - d) * sum_pr_pj

        print(Helper.dictToOrderedList(pr_pi, rev=True))

        pr = pr_pi
    with open('pr.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in pr.items():
            writer.writerow([k, v])
    return OrderedDict(Helper.dictToOrderedList(pr_pi, rev=True)[:5])


def main():
    global X,vec,terms,scoreArr

    test = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test')
    vec = TfidfVectorizer(stop_words=None, ngram_range=(1, 3))
    X = vec.fit_transform(test.values())
    terms = vec.get_feature_names()
    scoreArr = X.toarray()

    for file in test:
        buildGraphNew(test[file])
        break


main()
