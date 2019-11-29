import itertools
from collections import OrderedDict
from typing import List, Dict, Any

import networkx
from sklearn.feature_extraction.text import TfidfVectorizer

from project_p2_ex1 import Helper, buildGramsUpToN
from project_p2_ex2 import buildGraph, getInfo, computeWeightedPR


def calculateParameters(g: networkx.Graph, doc: str, scores: Dict[str, float], pr: Dict[str, float]):
    params = []

    max_cand_score = max(scores.values())
    all_cands = g.nodes
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
        params.append([cand_score, freq, cand_len, cand_term_count, first_match, last_match, spread, pr[cand]])
    return dict(zip(all_cands, params))


def getRecipRankFusionScore(words):
    RRFScore = {}
    for name, scores in words.items():
        wordScore = []
        for param in scores:
            wordScore.append(1 / (50 + param))
        RRFScore.update({name: sum(wordScore)})

    return Helper.dictToOrderedList(RRFScore, rev=True)[:5]


def main():
    test = Helper.getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test')
    test = dict(zip(list(test.keys())[:1], list(test.values())[:1]))
    info = getInfo()

    rrfScores = {}

    for i, doc_name in enumerate(test.keys()):
        g = buildGraph(test[doc_name], info['model'])
        pr = computeWeightedPR(g, i, info, n_iter=15)
        params = calculateParameters(g, test[doc_name], dict(zip(info['terms'], info['TF-IDF'][i])), pr)
        rrfScores[doc_name] = getRecipRankFusionScore(params)

    print('rrfScores', rrfScores)


if __name__ == '__main__':
    import time

    start = time.time()
    main()
    print(f'elapsed - {time.time() - start} seconds')
