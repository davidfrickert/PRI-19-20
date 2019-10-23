#!/bin/bash

# Florian Boudin (florian.boudin@univ-nantes.fr)
# 11 july 2017
# Script for preprocessing Inspec files

PATH_CORENLP=/nlp/stanford-corenlp-full-2016-10-31

################################################################################
# Prepare data
################################################################################
# unzip marujo-data.zip
################################################################################

################################################################################
# TEST split
################################################################################
# mkdir -p ../dataset

# ls dataset/*.txt > dataset.filelist

# java -cp "$PATH_CORENLP/*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP \
#      -annotators tokenize,ssplit,pos,lemma \
#      -outputDirectory dataset/ \
#      -ssplit.newlineIsSentenceBreak always \
#      -filelist dataset.filelist

# rm dataset.filelist

# for FILE in dataset/*.txt.xml
# do
#     mv $FILE ../${FILE%.txt.xml}.xml
# done
################################################################################

################################################################################
# TRAIN split
################################################################################
# mkdir -p ../train

# ls train/*.txt > train.filelist

# java -cp "$PATH_CORENLP/*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP \
#      -annotators tokenize,ssplit,pos,lemma \
#      -outputDirectory train/ \
#      -ssplit.newlineIsSentenceBreak always \
#      -filelist train.filelist

# rm train.filelist

# for FILE in train/*.txt.xml
# do
#     mv $FILE ../${FILE%.txt.xml}.xml
# done
################################################################################

################################################################################
# REFERENCES
################################################################################

# mkdir -p ../references

# python json_references.py dataset/ ../references/dataset.reader.stem.json key True
# python json_references.py train/ ../references/train.reader.stem.json key True

# python json_references.py dataset/ ../references/dataset.reader.json key False
# python json_references.py train/ ../references/train.reader.json key False


# ################################################################################
# # CLEAN UP
# ################################################################################
# rm -R dataset/
# rm -R train/
