from xml.dom.minidom import parse, parseString
import os

def main():
    directoryName = 'ake-datasets-master/datasets/500N-KPCrowd/train';
    directory = os.fsencode(directoryName)

    for f in os.listdir(directory):
        filePath = directoryName + '/' + f.decode("utf-8")
        datasource = open(filePath)
        dom = parse(datasource)

        sentences = dom.getElementsByTagName('sentence')
        sentence_list = []

        for i, sentence in enumerate(sentences):
            tokens = sentence.getElementsByTagName('token')
            sentence_list.append(' '.join([t.getElementsByTagName('word')[0].firstChild.nodeValue for t in tokens]))

        print(sentence_list)

main()
