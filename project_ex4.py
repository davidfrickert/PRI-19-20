from sklearn.linear_model import Perceptron

from project_ex2 import getDataFromDir
from project_ex3 import getAllCandidates


def createTargetList():
    return None

def calculateParameters(word):

p_classifier = Perceptron(alpha=0.1)

test = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test', mode='list')
allCandidatesTrain = getAllCandidates(test)
allCandidatesTest = getAllCandidates(test)

allParamsTrain = [] # lista de dicts ou dict k: doc_name, v: (palavra, params)
allParamsTest = []

for doc in allCandidatesTrain:
    for cand in doc:
        params = calculateParameters(cand)
        # inserir no sitio
    p_classifier.fit(allParamsTrain, createTargetList())

for doc in allCandidatesTest:
    for cand in doc:
        params = calculateParameters(cand)
    p_classifier.predict(allParamsTest)


# for dos documentos
# para cada doc extrair candidatos
# para cada candidato calcular os parâmetros

# for dos documentos
#   passamos a lista que contem os parametros de todos os candidatos
#   calculamos a lista de resultados [ 0 0 0 1 0 0 1 ]
#   fit
