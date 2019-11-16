import json
import operator
from collections import OrderedDict

import networkx
from networkx import pagerank
import nltk
import string
from nltk.corpus import stopwords
import itertools

class Helper:
    @staticmethod
    def dictToOrderedList(d: dict, rev=False):
        return sorted(d.items(), key=operator.itemgetter(1), reverse=rev)

def buildGramsUpToN(doc, n):
    ngrams = []
    doc = doc.lower()
    invalid = set(stopwords.words('english'))

    s = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in nltk.sent_tokenize(doc)]
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


def undirectedPR(g: networkx.Graph, n_iter=1, d=0.15):
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
    return OrderedDict(Helper.dictToOrderedList(pr_pi, rev=True)[:5])


def main():
    with open('nyt.txt', 'r') as doc:
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

        print('Nodes', g.nodes)

        # Step 3
        # Add edges
        # for each phrase (ngrams[i] is a list of words in phrase 'i') get all combinations of all words as edges
        [g.add_edges_from(itertools.combinations(ngrams[i], 2)) for i in range(len(ngrams))]

        # print('Edges', g.edges)

        pr = undirectedPR(g, n_iter=50)

        print('Final PR', pr)


main()
