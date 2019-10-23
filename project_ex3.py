import itertools
import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer

from project_ex2 import getDataFromDir, calcMetrics, merge


def getAllChunks(tagged_sents, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):

    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    chunker = nltk.chunk.regexp.RegexpParser(grammar)

    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))

    candidates = set([' '.join(word for word, pos, chunk in group).lower()
                      for key, group in itertools.groupby(all_chunks, lambda tpl: tpl[2] != 'O') if key])

    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]


def getTFIDFScore(dataset):
    # extract candidates from each text in texts, either chunks or words

    ds = [getAllChunks(text) for text in dataset.values()]

    vec = TfidfVectorizer(tokenizer=lambda i: i, lowercase=False)

    X = vec.fit_transform(ds).toarray()

    terms = vec.get_feature_names()

    return merge(dataset, terms, X)


def main():
    test = getDataFromDir('ake-datasets-master/datasets/500N-KPCrowd/test', mode='list')

    results = getTFIDFScore(test)

    print(results['politics_us-20782177'])

    calcMetrics(results, 'ake-datasets-master/datasets/500N-KPCrowd/references/test.reader.json')



if __name__ == '__main__':
    main()