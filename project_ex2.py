import string
import json
import os
from xml.dom.minidom import parse, parseString
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from os.path import  splitext

def average(dic):
    total = 0
    for item in dic:
        total += dic[item]
    
    return total / len(dic)

def calcTPFN(doc, reference_results, results, at = None):

    porter = PorterStemmer()
    true_positives = 0
    false_negatives = 0
    counter = 0

    
    non_relevant = results[doc][len(reference_results[doc]):]
    results[doc] = results[doc][:len(reference_results[doc])]

    if(at != None) :
        non_relevant = results[doc][at:]
        results[doc] = results[doc][:at]

    for term in reference_results[doc]:
        
        if at == counter:
            return true_positives, false_negatives

        for word in results[doc]:
            if porter.stem(word[0]) == term[0]:
                true_positives += 1
                break

        for word in non_relevant:
            if porter.stem(word[0]) == term[0]:
                false_negatives += 1
                break
        
        counter+=1

    return true_positives, false_negatives


def calcMetrics(results, reference):
    with open(reference) as f:
        reference_results = json.load(f)

        precision = {}
        recall = {}
        f1 = {}
        precision5 = {}
        non_relevant = {}

        for x in reference_results:

            true_positives, false_negatives = calcTPFN(x, reference_results, results)
            
            # TP / (TP + FP)
            # TP => corretos
            # FP => total(resultados) - corretos => incorretos

            precision[x] = float(true_positives) / float(true_positives + (len(reference_results[x]) - true_positives))

            # TP / (TP + FN)
            # TP => corretos
            # FN => quais keyphrases o nosso algoritmo identificou como não-relevante mas é relevante

            recall[x] = float(true_positives) / float(true_positives + false_negatives)
            if recall[x] or precision[x]:
                f1[x] = (2 * precision[x] * recall[x]) / (precision[x] + recall[x])
            else:
                f1[x] = 0.

        print("Precision: ")
        print(precision)
        print(average(precision), end="\n\n")
        print("Recall: ")
        print(recall)
        print(average(recall), end="\n\n")
        print("F1: ")
        print(f1)
        print(average(f1), end="\n\n")

        for x in reference_results:

            true_positives, false_negatives = calcTPFN(x, reference_results, results, 5)
            precision5[x] = float(true_positives) / float(true_positives + (len(reference_results[x]) - true_positives))

        print("Precision@5: ")
        print(precision5)
        print(average(precision5), end="\n\n")
        # calc precison, recall, f1 per doc & avg
        # avg precision@5 & avg precision


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


def merge(dataset, terms, scoreArr):
    data = {}
    for doc_index, doc_name in enumerate(dataset):
        doc_info = []
        for word_index, term in enumerate(terms):

            tf_idf = scoreArr[doc_index][word_index]
            if tf_idf != 0:
                doc_info.append((term, tf_idf * (len(term)/len(term.split(' ')))))

        # sort por tf_idf; elem = (term, tf_idf); elem[1] = tf_idf
        doc_info.sort(key=lambda elem: elem[1], reverse=True)

        data.update({doc_name: doc_info})
    return data

def getTFIDFScore(dataset):
    stopW = set(stopwords.words('english'))

    vec = TfidfVectorizer(stop_words=stopW, ngram_range=(1, 3), min_df=2)

    X = vec.fit_transform(dataset.values())

    terms = vec.get_feature_names()
    scoreArr = X.toarray()
    
    return merge(dataset, terms, scoreArr)

def main():
    test = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test')
    data = getTFIDFScore(test)
    calcMetrics(data, 'ake-datasets-master/datasets/500N-KPCrowd/references/test.reader.stem.json')


if __name__ == '__main__':
    main()