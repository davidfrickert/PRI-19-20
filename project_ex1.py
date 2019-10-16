from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

import nltk

def get_tfidf_score(doc):
    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    vec = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
    vec.fit(train.data)
    testdata = vec.transform(doc).sum(0)

    terms = vec.get_feature_names()

    data = []

    for col, term in enumerate(terms):
        if testdata[0, col] != 0:
            data.append((term, testdata[0, col] * len(term)))

    ranking = pd.DataFrame(data, columns=['term', 'rank'])
    print(ranking.sort_values('rank', ascending=False))


with open('text1.txt', 'r') as doc:
    # sent = ' '.join(doc.readlines()).lower()
    sent = nltk.sent_tokenize('\n'.join(doc.readlines()))


print(get_tfidf_score(sent))
