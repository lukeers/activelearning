from __future__ import division
import math
import nltk
import string
import os
import re
import collections
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import collections
from decimal import *
from itertools import islice
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

stop_words = set(stopwords.words('english'))

fullAnnot = "fullAnnotation.conf"
N = 10
vectorizer = CountVectorizer()
tokenize = lambda doc: doc.lower().split(" ")

def objectNames() :
  fullAnnotations = {}
  with open(fullAnnot,'r') as f:
   for line in f:
      l = line.split(",")
      l2 = line.replace(l[0]+ ",",'')
      l3 = l2.replace('\n','')
      fullAnnotations[l[0]] = l3
  return fullAnnotations

def getDocuments(fName):
#   fName = "3k_unfiltered_fulldataset.conf"

   instSentences = {}
   with open(fName, 'r') as f:
    for line in f:
     l = line.split(",")
     l2 = line.replace(l[0]+ ",",'')
     l3 = l2.replace('\n','')
     l3 = re.sub('[^A-Za-z0-9\ ]+', '', l3)
     l3 = l3.lower()
     if(line != "" and l3 != "") :
      if l[0] in instSentences.keys():
         sent = instSentences[l[0]]
         sent += " " + l3
         instSentences[l[0]] = sent
      else:
         instSentences[l[0]] = l3
   sortedinstSentences = collections.OrderedDict(sorted(instSentences.items()))  
   return sortedinstSentences


def fileringSentences(fName):
   with open(fName, 'r') as f:
    for line in f:
     l = line.split(",")
     l2 = line.replace(l[0]+ ",",'')
     l3 = l2.replace('\n','')
     l3 = re.sub('[^A-Za-z0-9\ ]+', '', l3)
     l3 = l3.lower()
     if(line != "" and l3 != "") :
       wLists = l3.split(" ")
       filtered_sentence = [w for w in wLists if not w in stop_words]
       filtered_sentence = [porter_stemmer.stem(w) for w in filtered_sentence]
       filtered_sentence = [lemmatizer.lemmatizer(w) for w in filtered_sentence]
       if len(filtered_sentence) > 0:
         print l[0]+ "," + " ".join(filtered_sentence)

def filterUniqWords(fName,allWords):
   with open(fName, 'r') as f:
    for line in f:
     l = line.split(",")
     l2 = line.replace(l[0]+ ",",'')
     l3 = l2.replace('\n','')
     l3 = re.sub('[^A-Za-z0-9\ ]+', '', l3)
     l3 = l3.lower()
     if(line != "" and l3 != "") :
       wLists = l3.split(" ")
       allWords.extend(wLists)
   return allWords

def sentenceToWordLists(docs):
   docLists = []
   for key in docs.keys():
      sent = docs[key]
      wLists = sent.split(" ")
      filtered_sentence = [w for w in wLists if not w in stop_words]
      docLists.append(filtered_sentence)
   return docLists

def sentenceToWordDicts(docs):
   docDicts = {}
   for key in docs.keys():
      sent = docs[key]
      wLists = sent.split(" ")
      docDicts[key] = wLists
   return docDicts

def findtfIDFLists(docLists):
   arWords = []
   for dList in docLists:
      arWords.extend(set(dList))
   arIDF = {}
   arIDFCount = Counter(arWords)
   for x in arIDFCount.keys():
      arIDF[x] = math.log(len(docLists)/arIDFCount[x])

   tdIDFLists = []
   for dList in docLists:
      dictC = Counter(dList)
      tfidfValues = []
      for word in dList:
         tfidfValues.append(dictC[word] * arIDF[word])
      tdIDFLists.append(tfidfValues)
   return tdIDFLists

def findTopNtfidfterms(docLists,tfidfLists,N):
   topTFIDFWordLists = []
   for i in range(len(docLists)):
      dList = docLists[i]
      tList = tfidfLists[i]
      dTFIDFMap = {}
      for j in range(len(dList)):
          dTFIDFMap[dList[j]] = tList[j]
      
      stC = sorted(dTFIDFMap.items(), key=lambda x: x[1])
      lastpairs = stC[len(stC) - N  :]
      vals = []
      for jj in lastpairs:
         vals.append(jj[0])
      topTFIDFWordLists.append(vals)
   return topTFIDFWordLists


def filterUniqDataPoints(fName,uDts,fl,fl1):
   with open(fName, 'r') as f:
    for line in f:
     l = line.split(",")
     l2 = line.replace(l[0]+ ",",'')
     l3 = l2.replace('\n','')
     l3 = re.sub('[^A-Za-z0-9\ ]+', '', l3)
     l3 = l3.lower()
     if(line != "" and l3 != "") :
       wLists = l3.split(" ")
       if fl == 1 :
          wLists =  [w for w in wLists if not w in stop_words]
       if fl1 == 1:
          wLists = [lemmatizer.lemmatize(w) for w in wLists]
       if l[0] in uDts.keys():
         ars = uDts[l[0]]
         ars.extend(wLists)
         ars = list(set(ars))
         uDts[l[0]] = ars
       else:
        uDts[l[0]] = list(set(wLists))
   return uDts

