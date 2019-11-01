import itertools
import json
import math

import nltk
from nltk import PorterStemmer
from sklearn.linear_model import Perceptron

from project_ex2 import getDataFromDir
from project_ex3 import getAllCandidates, getTFIDFScore, listOfTaggedToString, getBM25Score


def createTargetList(reference, term_list):
    target = {}
    with open(reference) as f:
        reference_results = json.load(f)
        for i, (name, doc_values) in enumerate(reference_results.items()):
            classes = []

            for term in term_list[i]:

                values = list(itertools.chain.from_iterable(doc_values))

                porter = PorterStemmer()
                s_term = porter.stem(term)

                if s_term in values:
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
            print(cand)
        else:
            cand_score = scores[cand] / max_cand_score

        cand_len = len(cand)
        cand_term_count = len(cand.split())

        words = nltk.pos_tag(nltk.word_tokenize(cand))
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

        params.append([cand_score, freq, cand_len, cand_term_count, first_match, last_match, spread, ne_cnt]) #cand_score,
    return params


def calcResults(predicted, true):
    TP = FP = TN = FN = 0
    for i, n in enumerate(predicted):
        if n == 1 and true[i] == 1:
            TP += 1
        elif n == 1 and true[i] == 0:
            FP += 1
        elif n == 0 and true[i] == 1:
            FN += 1
        elif n == 0 and true[i] == 0:
            TN += 1

    if TP or FP:
        precision = float(TP) / float(TP + FP)
    else:
        precision = 0

    if TP or FN:
        recall = float(TP) / float(TP + FN)
    else:
        recall = 0

    if recall or precision:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.

    print('prec')
    print(precision)
    print('rec')
    print(recall)
    print('f1')
    print(f1)

    return precision, recall, f1


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

bm25train = getBM25Score(train, mergetype='dict')
bm25test = getBM25Score(test, mergetype='dict')

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

for doc_index, doc_name in enumerate(test.keys()):
    params = calculateParameters(allCandidatesTest[doc_index], testStr[doc_index], bm25test[doc_name])

    predicted = p_classifier.predict(params)
    true = testTargets[doc_name]

    print('PERCEPTRON')
    print(predicted)
    print('REALITY')
    print(true)

    p, r, f = calcResults(predicted, true)

    precision.append(p)
    recall.append(r)
    f1.append(f)

print(sum(precision) / len(precision))
print(sum(recall) / len(precision))
print(sum(f1) / len(f1))

# for dos documentos
# para cada doc_name extrair candidatos
# para cada candidato calcular os parâmetros

# for dos documentos
#   passamos a lista que contem os parametros de todos os candidatos
#   calculamos a lista de resultados [ 0 0 0 1 0 0 1 ]
#   fit
