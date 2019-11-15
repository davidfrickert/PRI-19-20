import networkx
import nltk
import string
from nltk.corpus import stopwords
import itertools


def buildGramsUpToN(doc, n):
    ngrams = []
    doc = doc.lower()
    invalid = set(stopwords.words('english'))

    s = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in nltk.sent_tokenize(doc)]
    sents = [nltk.word_tokenize(_) for _ in s]

    # remove stop_words

    for sentence in sents:
        tormv = []
        for word in sentence:
            if word in invalid:
                tormv.append(word)
        [sentence.remove(w) for w in tormv]

    # end remove stop_words

    for sent in sents:
        ngram_list = []
        for N in range(1, n + 1):
            ngram_list += [sent[i:i + N] for i in range(len(sent) - N + 1)]
        ngrams.append([' '.join(gram) for gram in ngram_list])

    return ngrams


def main():
    with open('nyt.txt', 'r') as doc:
        # Step 1
        # Read document and build ngrams. Results is List of List of ngrams (each inner List is all ngrams of sentence)
        doc = ' '.join(doc.readlines())
        ngrams = buildGramsUpToN(doc, 3)

        print('Ngrams', ngrams)

        # Step 2
        # Build Graph and add all ngrams
        # from_iterable transforms [ [ng1, ng2], [ng3, ng4], ...] to [ ng1, ng2, ng3, ng4 ]
        # set removes any duplicate

        g = networkx.Graph()
        g.add_nodes_from(set(itertools.chain.from_iterable(ngrams)))

        print('Nodes', g.nodes)

        # Step 3
        # Add edges
        #
        [g.add_edges_from(itertools.combinations(ngrams[i], 2)) for i in range(len(ngrams))]

        # print('Edges', g.edges)


main()
