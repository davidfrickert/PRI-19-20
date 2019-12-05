# parse
import os
import xml.etree.cElementTree as et
from urllib.request import urlopen
from xml.etree.ElementTree import parse

# gen plot
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

# gen wordcloud
import pandas as pd
from nltk import sent_tokenize, pos_tag
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
# gen html
from yattag import Doc

# keyword gen
#parse
from urllib.request import urlopen
from xml.etree.ElementTree import parse
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.stem import PorterStemmer

#gen wordcloud
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#gen plot
import matplotlib.pyplot as plt

#gen html
from yattag import Doc
import xml.etree.cElementTree as et

#keyword gen
from project_p2_ex3 import run


class Helper:
    @staticmethod
    def printDict(d):
        for item in d:
            print(item, "-> ", d[item])


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
        documents[cat] += title+" "+description+" "
    
    return documents

def plotKeyphrases(category):

    df = pd.read_csv("CSV/" + category + ".csv")

    plt.figure(figsize=(15, 10))
    df.groupby("word").max().sort_values(by="weight", ascending=False)["weight"].plot.bar()
    plt.xticks(rotation=50)
    plt.xlabel("Word")
    plt.ylabel("Weight")
    # plt.show()
    plt.savefig(category + '1.png') #save to file


def generateWordCloud(documents, category):
    for item in documents:
        
        s = ""
        for word in documents[item]:
            s = s + word + " "

        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(s)
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

                    with tag('h1'):
                        text("Wordcloud:")

                    with tag('div', id='photo-container'):
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

            #print(sent)
            tags = pos_tag(' '.join(sent.split()).split(" "))
            #print(tags)
            for word in tags:
                token = et.SubElement(tokens, "token")
                token.set("id", str(wordCounter))
                et.SubElement(token, "lemma").text = porter.stem(word[0])
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
    f = open("CSV/"+category+".csv", "w+")
    s = ""
    for word in keywords:
        s += word + "," + str(keywords[word]) + "\n"
    
    f.write(s)
    f.close()

def fetchCategory(category: str):
    BASE_URL = 'https://rss.nytimes.com/services/xml/rss/nyt/'
    xml = getXML(f'{BASE_URL + category}.xml')
    documents = formatDocuments(xml, category)
    
    createNewsFiles(documents, category)
    
    keywords = run(f'news/{category}')
    return documents, keywords

def main():
    cats = ['Technology', 'World', 'US', 'HomePage', 'Politics']
    text_file = open("page.html", "a+")
    text_file.truncate(0)

    for category in cats:
        documents, keywords_with_score = fetchCategory(category)
        keywords = dict(zip(keywords_with_score.keys(),  [d.keys() for d in keywords_with_score.values()]))
        Helper.printDict(keywords_with_score)

        #keyword
        # word -> score
        
        #createCSV(keywords, category)

        #plotKeyphrases(category)
        
        generateWordCloud(keywords, category)
        
        text_file.write(generateHTML(keywords, category))

    text_file.close()


if __name__ == "__main__":
    main()
