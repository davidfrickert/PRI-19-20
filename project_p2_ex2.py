import csv
import itertools
import string
from collections import OrderedDict
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from platform import system
from time import time

import networkx
import nltk
from networkx import pagerank
from nltk.corpus import stopwords
from psutil import cpu_count
from sklearn.feature_extraction.text import TfidfVectorizer

from project_p2_ex1 import Helper
from gensim.models import KeyedVectors

stop_words = set(stopwords.words('english'))


def buildGramsUpToN(doc, n):
    ngrams = []
    doc = doc.lower()

    s = [sentence.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for sentence in
         nltk.sent_tokenize(doc)]
    sents = [nltk.word_tokenize(_) for _ in s]

    # remove stop_words

    for sentence in sents:
        tormv = []
        for word in sentence:
            if word in stop_words:
                tormv.append(word)
        [sentence.remove(w) for w in tormv]

    # end remove stop_words

    for sent in sents:
        ngram_list = []
        for N in range(1, n + 1):
            ngram_list += [sent[i:i + N] for i in range(len(sent) - N + 1)]
        ngrams.append([' '.join(gram) for gram in ngram_list])

    return ngrams


def buildGraph(doc, model, collection: list = None):
    ngrams = buildGramsUpToN(doc, 3)

    # print('Ngrams', ngrams)

    # Step 2
    # Build Graph and add all ngrams
    # from_iterable transforms [ [ng1, ng2], [ng3, ng4], ...] to [ ng1, ng2, ng3, ng4 ]
    # OrderedDict instead of set() for determinisic behaviour (every execution results in the same order in nodes)
    # deletes duplicates
    g = networkx.Graph()

    g.add_nodes_from(list(OrderedDict.fromkeys((itertools.chain.from_iterable(ngrams)))))

    # print('Nodes', g.nodes())

    # Step 3
    # Add edges
    # for each phrase (ngrams[i] is a list of words in phrase 'i') get all combinations of all words as edges
    edges = list(itertools.chain.from_iterable([itertools.combinations(ngrams[i], 2) for i in range(len(ngrams))]))
    new_edges = addWeights(edges, model)  # , collection=collection) #uncomment for co-occurence
    g.add_weighted_edges_from(new_edges)

    # print('Edges', g.edges)

    return g


# encontra o termo e retorna o seu score idf
'''
    @:arg doc_index: index of document to search in TF-IDF Matrix
    @:arg cand: candidate to search
    @:arg info: dict that has term list, TF-IDF Matrix and Word2Vec Model
'''


def prior(doc_index, cand, info):
    if cand in info['terms']:
        return info['TF-IDF'][doc_index][info['terms'].index(cand)]

    return 0


def priorCandLocation(cand: str, doc: str):
    # normalized location, inverted because sentences in first keyphrases
    # are more relevant
    nsw_doc = Helper.filterStopWords(doc)
    first_match = nsw_doc.find(cand) / len(nsw_doc)
    last_match = nsw_doc.rfind(cand) / len(nsw_doc)
    spread = last_match - first_match
    if spread != 0:
        return spread * len(cand)
    else:
        return first_match * len(cand)


def addWeights(edges, model, collection=None):
    # weights = co_ocurrence(edges, collection)
    weights = weightsWMD(edges, model)
    n_edges = [e + (w,) for (e, w) in zip(edges, weights)]
    return n_edges


def weightsWMD(edges, model: KeyedVectors):
    weights = [model.wmdistance(*e) for e in edges]
    weights = [1 - (w / max(weights)) for w in weights]
    return weights


def co_ocurrence(edges, collection):
    co_oc = []
    for doc in collection:
        sentences = nltk.sent_tokenize(doc)
        sentences = [sentence.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                     for sentence in sentences]
        new_sents = []
        for sent in sentences:
            new_sents.append(' '.join([word for word in nltk.word_tokenize(sent) if word not in stop_words]))
        sentences = new_sents

        for sent in sentences:
            for i, edge in enumerate(edges):
                not_in_vec = len(co_oc) < i + 1
                w0, w1 = edge
                co_oc_n = 0 if not_in_vec else co_oc[i]
                # if w0 == 'york 3':
                #     print('h')
                if w0 in sent and w1 in sent:
                    co_oc_n += 1
                if not_in_vec:
                    co_oc.append(co_oc_n)
                else:
                    co_oc[i] = co_oc_n
    return co_oc


def getKeyphrases(doc: str, info: dict, collection: list = None, doc_i=None):
    if not doc_i:
        doc_i = collection.index(doc)
    print(f'started graph for doc nº {doc_i}')
    g = buildGraph(doc, info['model'], collection=collection)
    # change n_iter, 1 for now for testing purposes
    pr = computeWeightedPR(g, doc_i, info, n_iter=15)
    kf = list(OrderedDict(Helper.dictToOrderedList(pr, rev=True)).keys())
    print(f'finished graph for doc nº {doc_i}')
    return kf, g


def computeWeightedPR(g: networkx.Graph, doc_i: int, info: dict, n_iter=1, d=0.15):
    pr = pagerank(g)
    diff_list = []

    # cand := pi, calculate PR(pi) for all nodes
    for _ in range(n_iter):
        pr_pi = {}
        for pi in g.nodes():
            pi_links = list(map(lambda e: e[1], g.edges(pi)))

            # sum_pr_pj = sum([pr[e[1]] / len(g.edges(e[1])) for e in pi_links])
            # todos os candidatos ou so os que estao ligados ao cand agora
            # ∑pjPrior(pj)
            rst_array = []
            for pj in pi_links:
                pj_links = list(map(lambda e: e[1], g.edges(pj)))
                bot = sum([g.edges[pj, pk]['weight'] for pk in pj_links])
                if bot == 0:
                    continue
                top = pr[pj] * g.edges[pj, pi]['weight']
                rst = top / bot
                rst_array.append(rst)
            rst = sum(rst_array)

            div = sum([prior(doc_i, w, info) for w in pi_links])
            # div = sum([priorCandLocation(w, doc) for w in pi_links])

            if div == 0:
                div = 1

            pr_pi[pi] = d * (prior(doc_i, pi, info) / div) + (1 - d) * rst
            # pr_pi[pi] = d * (priorCandLocation(pi, doc) / div) + (1 - d) * rst
            # pr_pi[cand] = d / N + (1 - d) * sum_pr_pj

        diff = Helper.getMaxDiff(pr.values(), pr_pi.values())
        sum_pr = sum(pr_pi.values())

        print(Helper.dictToOrderedList(pr_pi, rev=True))
        print('Max diff', diff)
        print('sum of PR', sum_pr)

        # pr = old iter, pr_pi = new iter
        diff_list.append(diff)
        pr = pr_pi
        # converges at max rate of 0.01
        if diff < 0.1:
            break
    # with open('pr.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     for k, v in pr.items():
    #         writer.writerow([k, v])
    return pr


def getInfo():
    model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
    #model.save('model.vec')
    #model = KeyedVectors.load('model.vec', mmap='r')
    terms, tf_idf = calculateTF_IDFAndNormalize('ake-datasets-master/datasets/500N-KPCrowd/test', 'ake-datasets-master/datasets/500N-KPCrowd/train')
    info = {'terms': terms, 'TF-IDF': tf_idf, 'model': model}
    return info


'''
    ds: dict[doc_name, doc]
'''


def single_process(ds):
    kfs = {}
    info = getInfo()
    # lower case our collection as list
    collection = list(map(lambda doc: doc.lower(), ds.values()))
    for i, file in enumerate(ds):
        d_pr = getKeyphrases(ds[file].lower(), info, doc_i=i, collection=collection)
        kfs.update({file: d_pr})

    meanAPre, meanPre, meanRe, meanF1 = Helper.results(kfs, 'ake-datasets-master/datasets/500N-KPCrowd/references'
                                                            '/test.reader.stem.json')
    print(f'Mean Avg Pre for {len(kfs.keys())} documents: ', meanAPre)
    print(f'Mean Precision for {len(kfs.keys())} documents: ', meanPre)
    print(f'Mean Recall for {len(kfs.keys())} documents: ', meanRe)
    print(f'Mean F1 for {len(kfs.keys())} documents: ', meanF1)


'''
    ds: dict[doc_name, doc]
'''


def multi_process(ds):
    info = getInfo()
    # cpu_count(logical=Helper.logical())
    with ProcessPoolExecutor(max_workers=cpu_count(logical=Helper.logical())) as executor:
        fts = {}
        kfs = {}
        collection = list(map(lambda doc: doc.lower(), ds.values()))

        for i, file in enumerate(ds):
            fts.update({executor.submit(getKeyphrases, ds[file].lower(), info, doc_i=i, collection=collection): file})
        for future in as_completed(fts):
            file = fts[future]
            kfs.update({file: future.result()})
    meanAPre, meanPre, meanRe, meanF1 = Helper.results(kfs,
                                                       'ake-datasets-master/datasets/500N-KPCrowd/references/test'
                                                       '.reader.stem.json')
    print(f'Mean Avg Pre for {len(kfs.keys())} documents: ', meanAPre)
    print(f'Mean Precision for {len(kfs.keys())} documents: ', meanPre)
    print(f'Mean Recall for {len(kfs.keys())} documents: ', meanRe)
    print(f'Mean F1 for {len(kfs.keys())} documents: ', meanF1)


def calculateTF_IDFAndNormalize(datasetFileName: str, backgroundCollectionFileName: str):
    ds = Helper.getDataFromDir(datasetFileName)
    extra = Helper.getDataFromDir(backgroundCollectionFileName)

    vec = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 3))
    vec.fit(extra.values())
    X = vec.fit_transform(ds.values())
    terms = vec.get_feature_names()
    tf_idf = X.toarray()

    new_tf_idf = [[doc[j] * (len(terms[j]) / len(terms[j].split())) for j in range(len(terms))] for doc in tf_idf]
    max_val = [max(tf_idf[i]) for i in range(len(tf_idf))]
    max_val = max(max_val)
    # normalize each row (document) with the respective max value
    #new_tf_idf = [[doc[j] / max_val[i] for j in range(len(terms))] for i, doc in enumerate(new_tf_idf)]
    new_tf_idf = [[doc[j] / max_val for j in range(len(terms))] for i, doc in enumerate(new_tf_idf)]
    return terms, new_tf_idf


def main():
    TEST = False
    test = Helper.getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test')
    # test = dict(zip(list(test.keys())[:2], list(test.values())[:2]))

    # model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
    # model = None
    # tfidf = {'terms': terms, 'scoreArr': scoreArr, 'model': model}
    # if windows and not do logical (checks v >= 3.8.0)
    if (system() == 'Windows' and not Helper.logical()) or TEST:  # True:#
        single_process(test)
    # if linux/mac or windows with v >= 3.8.0
    else:
        multi_process(test)


if __name__ == '__main__':
    start = time()
    main()
    print(time() - start)
