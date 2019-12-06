import itertools
from collections import OrderedDict
from typing import List, Dict, Any

import networkx
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from truecase import get_true_case
from project_ex3 import getAllCandidates, listOfTaggedToString
from project_p2_ex1 import Helper, buildGramsUpToN
from project_p2_ex2 import buildGraph, getInfo, computeWeightedPR, getPageRankOfDataset
import numpy as np


def calculateParameters(doc: str, scores: Dict[str, float], cands, pr: Dict[str, float] = None):
    params = []

    max_cand_score = max(scores.values())
    all_cands = cands
    for cand in all_cands:

        freq = doc.count(cand)

        # pagerank_score = pr[cand]

        if cand not in scores:
            cand_score = 0.
        else:
            cand_score = scores[cand] / max_cand_score

        cand_len = len(cand)
        cand_term_count = len(cand.split())

        first_match = doc.find(cand) / len(doc)
        last_match = doc.rfind(cand) / len(doc)
        ne_cand = get_true_case(cand)
        words = nltk.pos_tag(nltk.word_tokenize(ne_cand))
        ne = nltk.tree2conlltags(nltk.ne_chunk(words))
        ne = [' '.join(word for word, pos, chunk in group).lower()
              for key, group in itertools.groupby(ne, lambda tpl: tpl[2] != 'O') if key]

        ne_cnt = len(ne[0].split()) if ne else 0

        if first_match == last_match:
            spread = 0.
        else:
            spread = last_match - first_match

        params.append(
            [cand_score, cand_len, cand_term_count, first_match, 1 - last_match, ne_cnt])#, pagerank_score])  # , r[cand]])

    params = np.array(params)
    max_ = params.max(axis=0)
    params = np.divide(params, max_, out=np.zeros_like(params), where=max_ != 0)
    return dict(zip(all_cands, params))

def getRecipRankFusionScore(words):
    RRFScore = {}
    for name, scores in words.items():
        wordScore = []
        for param in scores:
            wordScore.append(1 / (50 + param))
        RRFScore.update({name: sum(wordScore)})

    return dict(Helper.dictToOrderedList(RRFScore, rev=True)[:50])


def run(path):
    test = Helper.getDataFromDir(path, mode='list')
    testStr = listOfTaggedToString(test)
    print(test.keys())
    # return
    # test = dict(zip(list(test.keys()), list(test.values())))
    info = getInfo(test)
    cands = getAllCandidates(test, deliver_as='sentences')
    rrfScores = {}

    for i, doc_name in enumerate(test.keys()):
        # cands =  buildGramsUpToN(test[doc_name], 3)
        # g = buildGraph(cands, info['model'])
        # pr = computeWeightedPR(g, i, info, n_iter=15)
        params = calculateParameters(testStr[i], info['score'][doc_name],
                                     list(itertools.chain.from_iterable(cands[i])))  # , pr)
        print(params)
        rrfScores[doc_name] = getRecipRankFusionScore(params)

    return rrfScores


def main():
    test = Helper.getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test', mode='list')
    testStr = listOfTaggedToString(test)
    # test = dict(zip(list(test.keys()), list(test.values())))
    info = getInfo(test)
    cands = getAllCandidates(test, deliver_as='sentences')
    #kfs = getPageRankOfDataset(test)
    rrfScores = {}

    for i, doc_name in enumerate(test.keys()):
        # cands =  buildGramsUpToN(test[doc_name], 3)
        # g = buildGraph(cands, info['model'])
        # pr = computeWeightedPR(g, i, info, n_iter=15)
        params = calculateParameters(testStr[i], info['score'][doc_name], list(itertools.chain.from_iterable(cands[i])),
                                     pr=None)#kfs[doc_name])  # , pr)
        print(params)
        rrfScores[doc_name] = list(getRecipRankFusionScore(params).keys())

    print('rrfScores', rrfScores)
    meanAPre, meanPre, meanRe, meanF1 = Helper.results(rrfScores, 'ake-datasets-master/datasets/500N-KPCrowd/references'
                                                                  '/test.reader.stem.json')
    print(f'Mean Avg Pre for {len(rrfScores.keys())} documents: ', meanAPre)
    print(f'Mean Precision for {len(rrfScores.keys())} documents: ', meanPre)
    print(f'Mean Recall for {len(rrfScores.keys())} documents: ', meanRe)
    print(f'Mean F1 for {len(rrfScores.keys())} documents: ', meanF1)


if __name__ == '__main__':
    import time

    start = time.time()
    main()
    print(f'elapsed - {time.time() - start} seconds')
