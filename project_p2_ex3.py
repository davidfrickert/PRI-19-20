import itertools
import networkx
from networkx import pagerank
from sklearn.feature_extraction.text import TfidfVectorizer

from project_ex1 import Helper, getKeyphrasesFromGraph, buildGramsUpToN
from collections import OrderedDict


def calculateParameters(g: networkx.Graph, all_cands, doc, scores):
    params = []
    #graphScores = pagerank(g)
    max_cand_score = max(scores.values())

    for cand in all_cands:

        freq = doc.count(cand)

        if cand not in scores:
            cand_score = 0.
        else:
            cand_score = scores[cand] / max_cand_score

        cand_len = len(cand)
        cand_term_count = len(cand.split())

        first_match = doc.find(cand) / len(doc)
        last_match = doc.rfind(cand) / len(doc)

        if first_match == last_match:
            spread = 0.
        else:
            spread = last_match - first_match
        params.append([cand_score, freq, cand_len, cand_term_count, first_match, last_match, spread, graphScores[cand]])
    return dict(zip(all_cands, params))


def getRecipRankFusionScore(words):
    RRFScore = {}
    for name, scores in words.items():
        wordScore = []
        for param in scores:
            wordScore.append(1 / (50 + param))
        RRFScore.update({name: sum(wordScore)})

    return Helper.dictToOrderedList(RRFScore, rev=True)[:5]


def buildGraph(doc):
    nGrams = buildGramsUpToN(doc, 3)

    g = networkx.Graph()

    g.add_nodes_from(list(OrderedDict.fromkeys((itertools.chain.from_iterable(nGrams)))))

    [g.add_edges_from(itertools.combinations(nGrams[i], 2)) for i in range(len(nGrams))]

    return g


def main():
    test = Helper.getDataFromDir(r'C:\Program Files (x86)\PycharmProjects\Projeto_PRI_Parte2\500N-KPCrowd/test')

    vec = TfidfVectorizer(stop_words=None, ngram_range=(1, 3))
    X = vec.fit_transform(test.values())
    terms = vec.get_feature_names()
    scoreArr = X.toarray()
    tfidf = {'terms': terms, 'scoreArr': scoreArr}
    rrfScores = {}
    for i, doc_name in enumerate(test.keys()):
        g = buildGraph(test[doc_name])
        params = calculateParameters(g, getKeyphrasesFromGraph(g, n_iter=1), test[doc_name], dict(zip(terms, scoreArr[i])))
        rrfScores[doc_name] = getRecipRankFusionScore(params)

    print(rrfScores)


if __name__ == '__main__':
    import time

    start = time.time()
    main()
    print(time.time() - start)
