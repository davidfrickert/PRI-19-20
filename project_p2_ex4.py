# parse
import itertools
import os

# gen plot
from collections import OrderedDict

import matplotlib

from project_p2_ex1 import Helper
from project_p2_ex2 import getPageRankOfDataset

matplotlib.use('agg')

# keyword gen
# parse
from urllib.request import urlopen
from xml.etree.ElementTree import parse
from nltk import sent_tokenize, pos_tag
from nltk.stem import PorterStemmer

# gen wordcloud
import pandas as pd
from wordcloud import WordCloud

# gen plot
import matplotlib.pyplot as plt

# gen html
from yattag import Doc
import xml.etree.cElementTree as et

# keyword gen
from project_p2_ex3 import run


def getXML(url):
    open_url = urlopen(url)
    xml = parse(open_url)
    return xml


def formatDocuments(xml, cat):
    documents = dict()

    documents[cat] = ""
    for item in xml.iterfind('channel/item'):
        title = item.findtext('title')
        description = item.findtext('description')
        documents[cat] += title + " " + description + " "

    return documents


def plotKeyphrases(category):
    df = pd.read_csv("CSV/" + category + ".csv")

    plt.figure(figsize=(10, 7.5))
    df.groupby("word").max().sort_values(by="weight", ascending=False)["weight"].plot.bar()
    rot = 90 if category == 'Global' else 50
    plt.xticks(rotation=rot)
    plt.xlabel("Word")
    plt.ylabel("Weight")
    plt.tight_layout()
    #plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig("img/" + category + '1.png')  # save to file


def generateWordCloud(documents, category):
    for item in documents:

        if category == "Global":
            tmp = dict(itertools.islice(documents[item].items(), 29))
        else:
            tmp = dict(itertools.islice(documents[item].items(), 9))

        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(tmp)
        path = 'img/'
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + category + ".png"
        wordcloud.to_file(path)


def generateHTML(documents, category):
    doc, tag, text = Doc().tagtext()

    with tag('html'):
        with tag('body'):
            for document in documents:
                with tag('div', id='news', align='center'):
                    with tag('h1'):
                        text("Category:")

                    text(category)

                    with tag('h1'):
                        text("Keywords:")

                    s = ""
                    for word in documents[document]:
                        s = s + word + "; "
                    text(s)

                    # with tag('h1'):
                    # text("Wordcloud:")

                    with tag('div', id='photo-container'):
                        path = "img/" + category + "1.png"
                        doc.stag('img', src=path, klass="photo")
                        with tag('br'):
                            pass
                        path = "img/" + category + ".png"
                        doc.stag('img', src=path, klass="photo")

                with tag('br'):
                    pass

    return doc.getvalue()


def createNewsFiles(documents, category: str):
    porter = PorterStemmer()

    root = et.Element("root")
    document = et.SubElement(root, "document")

    for i, title in enumerate(documents):
        sentenceCounter = 1
        wordCounter = 1

        sentences = et.SubElement(document, "sentences")
        for sent in sent_tokenize(documents[title]):
            sentence = et.SubElement(sentences, "sentence")
            sentence.set("id", str(sentenceCounter))
            tokens = et.SubElement(sentence, "tokens")

            # print(sent)
            tags = pos_tag(' '.join(sent.split()).split(" "))
            # print(tags)
            for word in tags:
                l = word[0].replace(',', '').replace('.', '').replace('?', '').replace(
                    '!', '')
                if not l:
                    continue
                token = et.SubElement(tokens, "token")
                token.set("id", str(wordCounter))

                et.SubElement(token, "lemma").text = l

                et.SubElement(token, "POS").text = word[1]

                wordCounter += 1

            sentenceCounter += 1

    tree = et.ElementTree(root)
    # tree.write("news/" + title.replace(" ", "_") + ".xml")

    path = f'news/{category}/'

    if not os.path.exists(path):
        os.makedirs(path)

    tree.write(path + str(i) + ".xml")


def createCSV(keywords, category):
    path = "CSV/"
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(path + category + ".csv", "w+", encoding='utf8')
    s = "word,weight\n"
    if category == "Global":
        limit = 29
    else:
        limit = 9

    for word in keywords:
        for i, w in enumerate(keywords[word]):
            s += w + "," + str(keywords[word][w]) + "\n"
            if i == limit:
                break

    f.write(s)
    f.close()


def fetchCategory(category: str):
    BASE_URL = 'https://rss.nytimes.com/services/xml/rss/nyt/'
    xml = getXML(f'{BASE_URL + category}.xml')
    documents = formatDocuments(xml, category)

    return documents


def calcKeywords(documents, category):
    createNewsFiles(documents, category)

    # keywords = run(f'news/{category}')

    keywords = getPageRankOfDataset(f'news/{category}')
    keywords = {'0': OrderedDict(Helper.dictToOrderedList(keywords['0'], rev=True))}
    return keywords


def relevantInCategory(category, categoryScores, globalScores):
    doc = '0'
    cScores = categoryScores[doc]
    gScores = globalScores[doc]
    relevants = {doc : {}}
    for kf, score in cScores.items():
        # if score higher in category than global, then relevant, else not
        if cScores[kf] > gScores[kf]:
            relevants.append(cScores)
            print(kf + ' relevant in ' + category)
        else:
            print(kf + ' not relevant in ' + category)

    return relevants


def main():
    cats = ['Technology', 'Politics', 'Sports'
            # 'World', 'US', 'HomePage','Business', 'Economy', 'Soccer', 'Science', 'Environment', 'Travel', 'Arts', 'Books'
            ]
    text_file = open("page.html", "a+")
    text_file.truncate(0)

    global_doc = ''
    # build global doc
    for category in cats:
        documents = fetchCategory(category)
        global_doc += ' '.join(list(documents.values()))

    gCat = 'Global'
    results = calcKeywords({'0': global_doc}, gCat)
    kf = dict(zip(results.keys(), [d.keys() for d in results.values()]))
    createCSV(results, gCat)
    plotKeyphrases(gCat)
    generateWordCloud(results, gCat)
    text_file.write(generateHTML(kf, gCat))

    print(results)
    for category in cats:
        documents = fetchCategory(category)
        keywords_with_score = calcKeywords(documents, category)
        keywords = dict(zip(keywords_with_score.keys(), [d.keys() for d in keywords_with_score.values()]))
        Helper.printDict(keywords_with_score)

        # keyword
        # word -> score

        createCSV(keywords_with_score, category)

        plotKeyphrases(category)

        generateWordCloud(keywords_with_score, category)

        text_file.write(generateHTML(keywords, category))

        #relevantInCategory(category, keywords_with_score, results)

    text_file.close()


if __name__ == "__main__":
    main()
