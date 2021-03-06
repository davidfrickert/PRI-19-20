import itertools
import json
import math
import pprint
import time

import nltk
from nltk import PorterStemmer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_score, f1_score, recall_score, average_precision_score
from truecase import get_true_case

from project_ex2 import getDataFromDir
from project_ex3 import getAllCandidates, getTFIDFScore, listOfTaggedToString, getBM25Score
import sklearn.metrics

# nltk.download('maxent_ne_chunker')
# nltk.download('words')
from project_p2_ex1 import Metrics, Helper


def createTargetList(reference, term_list):
    target = {}
    with open(reference) as f:
        reference_results = json.load(f)
        for i, (name, doc_values) in enumerate(reference_results.items()):
            classes = []

            for term in term_list[i]:

                values = list(itertools.chain.from_iterable(doc_values))

                porter = PorterStemmer()

                stemmed = ""

                for w in term.split():
                    stemmed += porter.stem(w) + " "
                stemmed = stemmed[:-1]

                # s_term = porter.stem(term)

                if stemmed in values:
                    classes.append(1)
                else:
                    classes.append(0)

            target.update({name: classes})
    return target


def calculateParameters(all_cands, doc, scores):
    params = []

    max_cand_score = max(scores.values())

    for cand in all_cands:

        freq = doc.count(cand)

        if cand not in scores:
            cand_score = 0.
        else:
            cand_score = scores[cand] # / max_cand_score

        cand_len = len(cand)
        cand_term_count = len(cand.split())
        ne_cand = get_true_case(cand)
        words = nltk.pos_tag(nltk.word_tokenize(ne_cand))
        ne = nltk.tree2conlltags(nltk.ne_chunk(words))
        ne = [' '.join(word for word, pos, chunk in group).lower()
              for key, group in itertools.groupby(ne, lambda tpl: tpl[2] != 'O') if key]

        ne_cnt = len(ne[0].split()) if ne else 0

        first_match = doc.find(cand) / len(doc)
        last_match = doc.rfind(cand) / len(doc)

        # if cand_term_count == 1:
        #     cohesion = 0.
        # else:
        #     cohesion = cand_term_count * (1 + math.log(freq, 10)) * freq /

        if first_match == last_match:
            spread = 0.
        else:
            spread = last_match - first_match

        # print([cand_score, freq, cand_len, cand_term_count, first_match, last_match, spread, ne_cnt])

        params.append([cand_score, cand_len, cand_term_count, first_match, last_match, spread, ne_cnt]) #cand_score,
    return params


def calcResults(predicted, true):
    # , average_precision_score(true, predicted)
    return precision_score(true, predicted), recall_score(true, predicted), f1_score(true, predicted)

p_classifier = Perceptron(alpha=0.1)

train = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/train', mode='list')
trainStr = listOfTaggedToString(train)

test = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test', mode='list')
testStr = listOfTaggedToString(test)

allCandidatesTrain = getAllCandidates(train)
allCandidatesTest = getAllCandidates(test)


# bm25
# 0.3558736870896098
# 0.7640337163696295
# 0.4607649659785287

# TF IDF
# 0.37863851957992073
# 0.31571002226187983
# 0.3159382700815522

bm25train = getBM25Score(train, mergetype='dict', min_df=1)
bm25test = getBM25Score(test, mergetype='dict', min_df=1)

targets = createTargetList('ake-datasets-master/datasets/500N-KPCrowd/references/train.reader.stem.json',
                           allCandidatesTrain)

testTargets = createTargetList('ake-datasets-master/datasets/500N-KPCrowd/references/test.reader.stem.json',
                               allCandidatesTest)

for doc_index, doc_name in enumerate(train.keys()):
    allParams = calculateParameters(allCandidatesTrain[doc_index], trainStr[doc_index], bm25train[doc_name])
    if not targets[doc_name].count(0) == len(targets[doc_name]):
        p_classifier.fit(allParams, targets[doc_name])

print('predict')

precision = []
recall = []
f1 = []
ap = []
tr = Helper.getTrueKeyphrases('ake-datasets-master/datasets/500N-KPCrowd/references/test.reader.stem.json')
kfs = {}

for doc_index, doc_name in enumerate(test.keys()):
    params = calculateParameters(allCandidatesTest[doc_index], testStr[doc_index], bm25test[doc_name])

    predicted = p_classifier.predict(params)
    plane = p_classifier.decision_function(params)
    true = testTargets[doc_name]

    print('PERCEPTRON')
    print(predicted)
    print('[P2]', plane)
    print('REALITY')
    print(true)

    rnk = {list(bm25test[doc_name].keys())[i]: v for i, v in enumerate(plane) if v > 0}
    rnk = list(dict(Helper.dictToOrderedList(rnk, rev=True)).keys())
    p, r, f = calcResults(predicted, true)
    kfs[doc_name] = rnk

    #precision.append(p)
    #recall.append(r)
    #f1.append(f)
    #ap.append(aps)

meanAPre, meanPre, meanRe, meanF1 = Helper.results(kfs, 'ake-datasets-master/datasets/500N-KPCrowd/references/test'
                                                        '.reader.stem.json')
print('--RESULTS--')
print('Precision = ', meanPre)
print('Recall = ', meanRe)
print('F1 = ', meanF1)
print('Mean AVG Precision = ', meanAPre)

# for dos documentos
# para cada doc_name extrair candidatos
# para cada candidato calcular os parâmetros

# for dos documentos
#   passamos a lista que contem os parametros de todos os candidatos
#   calculamos a lista de resultados [ 0 0 0 1 0 0 1 ]
#   fit
