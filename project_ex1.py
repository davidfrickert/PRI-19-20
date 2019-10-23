from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

import nltk

def get_tfidf_score(doc):
    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    vec = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
    vec.fit(train.data)

    tf_idf_matrix = vec.transform(doc)

    terms = vec.get_feature_names()

    data = []

    for col, term in enumerate(terms):
        if tf_idf_matrix[0, col] != 0:
            data.append((term, tf_idf_matrix[0, col] * (len(term) / len(term.split()))))

    ranking = pd.DataFrame(data, columns=['term', 'rank'])
    print(ranking.sort_values('rank', ascending=False).to_string())


with open('text1.txt', 'r') as doc:
    sent = [' '.join(doc.readlines())]
    print(sent)


print(get_tfidf_score(sent))
