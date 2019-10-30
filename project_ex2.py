import string
import json
import os
from xml.dom.minidom import parse, parseString
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from os.path import splitext


def average(dic):
    total = 0
    for item in dic:
        total += dic[item]

    return total / len(dic)


def calcTPFNFP(doc, reference_results, results):
    # print("\n", doc)
    porter = PorterStemmer()
    true_positives = 0
    false_negatives = 0
    false_positives = 0

    results[doc] = results[doc][:len(reference_results[doc])]

    for word in results[doc]:
        flag = 0

        stemed = ""

        for w in word[0].split(' '):
            stemed += porter.stem(w) + " "

        stemed = stemed[:-1]

        for term in reference_results[doc]:
            if stemed == term[0]:
                true_positives += 1
                flag = 1
                break

        if flag == 0:
            false_positives += 1

    for term in reference_results[doc]:
        flag = 0

        for word in results[doc]:

            stemed = ""

            for w in word[0].split(' '):
                stemed += porter.stem(w) + " "

            stemed = stemed[:-1]

            if stemed == term[0]:
                flag = 1
                break

        if flag == 0:
            false_negatives += 1

    return true_positives, false_negatives, false_positives


def precisionAt(doc, reference_results, results, at):
    porter = PorterStemmer()
    true_positives = 0
    counter = 0

    tmp = results[doc][:at]

    for word in tmp:

        stemed = ""

        for w in word[0].split(' '):
            stemed += porter.stem(w) + " "

        stemed = stemed[:-1]

        for term in reference_results[doc]:

            if stemed == term[0]:
                true_positives += 1
                break
    
        counter+=1

    return float(true_positives)/float(at)


def calcMetrics(results, reference):
    with open(reference) as f:
        reference_results = json.load(f)

        precision = {}
        recall = {}
        f1 = {}
        precision5 = {}

        for x in reference_results:

            true_positives, false_negatives, false_positives = calcTPFNFP(x, reference_results, results)

            precision[x] = float(true_positives) / float(true_positives + false_positives)

            recall[x] = float(true_positives) / float(true_positives + false_negatives)
            if recall[x] or precision[x]:
                f1[x] = (2 * precision[x] * recall[x]) / (precision[x] + recall[x])
            else:
                f1[x] = 0.

            precision5[x] = precisionAt(x, reference_results, results,5)


        print("Precision: ")
        print(precision)
        print(average(precision), end="\n\n")
        print("Recall: ")
        print(recall)
        print(average(recall), end="\n\n")
        print("F1: ")
        print(f1)
        print(average(f1), end="\n\n")
        print("Precision@5: ")
        print(precision5)
        print(average(precision5), end="\n\n")


def convertXML(xml):
    result = ""

    for i, sentence in enumerate(xml):
        tokens = sentence.getElementsByTagName('token')
        result += ' '.join([t.getElementsByTagName('lemma')[0].firstChild.nodeValue for t in tokens])
        result += ' '

    return result


def convertXMLToTaggedSents(xml):
    result = []

    for i, sentence in enumerate(xml):
        tokens = sentence.getElementsByTagName('token')
        result.append([(t.getElementsByTagName('lemma')[0].firstChild.nodeValue,
                        t.getElementsByTagName('POS')[0].firstChild.nodeValue) for t in tokens])
    return result


def getDataFromDir(path, mode='string'):
    directory = os.fsencode(path)

    docs = {}
    files = os.listdir(directory)
    files.sort()

    for f in files:

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
                doc_info.append((term, tf_idf * (len(term) / len(term.split(' ')))))

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
