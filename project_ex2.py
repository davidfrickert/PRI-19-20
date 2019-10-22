from xml.dom.minidom import parse, parseString
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import os
from os.path import splitext
import json

from sklearn.metrics import *


def calcMetrics(results, reference):
    with open(reference) as f:
        reference_results = json.load(f)

        # print(reference_results['politics_world-20786243'])

        # for x in reference_results['politics_world-20786243']:
        #    for term in results['politics_world-20786243']:

        #        if term[0] == x[0]:
        #            print(term, "=", x)

        precision = {}
        recall = {}
        f1 = {}
        non_relevant = {}

        for x in reference_results:
            true_positives = 0
            false_negatives = 0

            non_relevant[x] = results[x][len(reference_results[x]):]
            results[x] = results[x][:len(reference_results[x])]

            for term in reference_results[x]:
                for word in results[x]:
                    if word[0] == term[0]:
                        true_positives += 1
                        break

                for word in non_relevant[x]:
                    if word[0] == term[0]:
                        false_negatives += 1
                        break

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

        print(precision)
        print(recall)
        print(f1)


def convertXML(xml):
    result = ""

    for i, sentence in enumerate(xml):
        tokens = sentence.getElementsByTagName('token')
        result += ' '.join([t.getElementsByTagName('word')[0].firstChild.nodeValue for t in tokens])

    return result


def getDataFromDir(path):
    directory = os.fsencode(path)

    docs = {}

    for f in os.listdir(directory):
        filePath = path + '/' + f.decode("utf-8")

        with open(filePath) as datasource:
            dom = parse(datasource)
            xml = dom.getElementsByTagName('sentence')

            doc_name = splitext(f)[0].decode("utf-8")
            docs.update({doc_name: convertXML(xml)})

    return docs


def main():
    train = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/train')

    stopW = set(stopwords.words('english'))

    vec = TfidfVectorizer(stop_words=stopW, ngram_range=(1, 3))

    test = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test')

    vec.fit(train.values())

    X = vec.transform(test.values())

    terms = vec.get_feature_names()
    scoreArr = X.toarray()

    data = {}

    for doc_index, doc_name in enumerate(test):
        doc_info = []
        for word_index, term in enumerate(terms):

            tf_idf = scoreArr[doc_index][word_index]
            if tf_idf != 0:
                doc_info.append((term, tf_idf))

        # sort por tf_idf; elem = (term, tf_idf); elem[1] = tf_idf
        doc_info.sort(key=lambda elem: elem[1], reverse=True)

        # apenas queremos o top 5 (índices 0 a 5 (ñ inclusive))
        data.update({doc_name: doc_info})

    # print(len(data))

    calcMetrics(data, 'ake-datasets-master/datasets/500N-KPCrowd/references/test.reader.json')


if __name__ == '__main__':
    main()
