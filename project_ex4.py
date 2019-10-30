import itertools
import json
import re
import time

from nltk import PorterStemmer
from sklearn.linear_model import Perceptron

from project_ex2 import getDataFromDir
from project_ex3 import getAllCandidates, getBM25Score


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
    # print(target['art_and_culture-20880868'])
    return target


def calculateParameters(all_cands, doc, scores):
    params = []
    for cand in all_cands:
        # p = re.compile(r'\b' + cand + r'\b')
        freq = doc.count(cand)

        for bm25score in scores:
            if bm25score[0] == cand:
                cand_score = bm25score[1]
            else:
                cand_score = 0

        cand_len = len(cand)
        cand_term_count = len(cand.split())
        # first_match = p.search(doc.lower())

        # # first_occurrence = first_match.start() / len(doc)
        #
        # if cand_term_count == 1:
        #     spread = 0.0
        # #    last_occurrence = first_occurrence
        # else:
        #     #
        #     # match = ''
        #     #
        #     # for match in p.finditer(doc):
        #     #     pass
        #
        #     last_occurrence = doc.rfind(cand) / len(doc)
        #     spread = last_occurrence - first_occurrence

        params.append([freq, cand_len, cand_term_count])

    return params


p_classifier = Perceptron(alpha=0.1)

train = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/train', mode='list')
trainStr = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/train')

test = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test', mode='list')
testStr = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test')

allCandidatesTrain = getAllCandidates(train)
allCandidatesTest = getAllCandidates(test)

bm25train = getBM25Score(train)
bm25test = getBM25Score(test)

targets = createTargetList('ake-datasets-master/datasets/500N-KPCrowd/references/train.reader.stem.json',
                           allCandidatesTrain)

for doc_index, doc_name in enumerate(train.keys()):
    allParams = calculateParameters(allCandidatesTrain[doc_index], trainStr[doc_name].lower(), bm25train[doc_name])
    if not targets[doc_name].count(0) == len(targets[doc_name]):
        p_classifier.fit(allParams, targets[doc_name])

print('predict')

for doc_index, doc_name in enumerate(test.keys()):
    allParams = []

    params = calculateParameters(allCandidatesTest[doc_index], testStr[doc_name].lower(), bm25test[doc_name])

    print(p_classifier.predict(params))

# for dos documentos
# para cada doc_name extrair candidatos
# para cada candidato calcular os parâmetros

# for dos documentos
#   passamos a lista que contem os parametros de todos os candidatos
#   calculamos a lista de resultados [ 0 0 0 1 0 0 1 ]
#   fit
