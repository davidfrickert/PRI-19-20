import xml.dom.minidom
import os

xml = xml.dom.minidom.parse("1005058.xml")
sentences = xml.getElementsByTagName('sentence')

sentence_list = []

for i, sentence in enumerate(sentences):
    tokens = sentence.getElementsByTagName('token')
    sentence_list.append(' '.join([t.getElementsByTagName('word')[0].firstChild.nodeValue for t in tokens]))

print(sentence_list)
