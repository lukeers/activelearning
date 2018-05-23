import math
from collections import Counter
import collections
import util
import gensim, logging
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from scipy import spatial

# numpy
import numpy

# random
from random import shuffle

# classifier
from sklearn.linear_model import LogisticRegression

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

N = 5

class LabeledLineSentence(object):
    def __init__(self,docLists,docLabels):
        self.docLists = docLists
        self.docLabels = docLabels

    def __iter__(self):
        for index, arDoc in enumerate(self.docLists):
            yield LabeledSentence(arDoc, [self.docLabels[index]])

    def to_array(self):
        self.sentences = []
        for index, arDoc in enumerate(self.docLists):
            self.sentences.append(LabeledSentence(arDoc, [self.docLabels[index]]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

def square_rooted(x):
    return round(math.sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
   numerator = sum(a*b for a,b in zip(x,y))
   denominator = square_rooted(x)*square_rooted(y)
   return round(numerator/float(denominator),3)
 

oNames = util.objectNames()
objNames = collections.OrderedDict(sorted(oNames.items()))

docs = util.getDocuments()
docLabels = []
docNames = docs.keys()
for key in docs.keys():
   ar = key.split("/")
   docLabels.append(ar[1])
docLists = util.sentenceToWordLists(docs)
docDicts = util.sentenceToWordDicts(docs)
sentences = LabeledLineSentence(docLists,docLabels)
model = Doc2Vec(min_count=1, window=10, size=2000, sample=1e-4, negative=5, workers=8)

model.build_vocab(sentences.to_array())
token_count = sum([len(sentence) for sentence in sentences])
for epoch in range(10):
    model.train(sentences.sentences_perm(),total_examples = token_count,epochs=model.iter)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    model.train(sentences.sentences_perm(),total_examples = token_count,epochs=model.iter)

tfidfLists = util.findtfIDFLists(docLists)
topTFIDFWordLists = util.findTopNtfidfterms(docLists,tfidfLists,N)

#model.most_similar('arch')
#print model.docvecs.most_similar(docLabels[0])
degreeMap = {}
posNegMap = {}
negLabelDicts = {}
angles = []
for i , item1 in enumerate(docLabels):
  fDoc = model.docvecs[docLabels[i]]

  #print docNames[i] + "(" + objNames[docLabels[i]]+ ")" + ":",
  negPoints = []
  negLabels = set()
  cInstMap = {}
  cInstance = objNames[docLabels[i]] +  "(" + docLabels[i] + ")"
  for j,item2 in enumerate(docLabels):
     tDoc = model.docvecs[docLabels[j]]

     #cosineVal = 1 - spatial.distance.cosine(fDoc,tDoc)
     cosineVal = cosine_similarity(fDoc,tDoc)
     tInstance = objNames[docLabels[j]] +  "(" + docLabels[j] + ")"
     cValue = math.degrees(math.acos(cosineVal))
     cInstMap[tInstance] = cValue
     angles.append(cValue)
     if i != j :
       if cosineVal < 0.15:
         #print docNames[j] + ",",
         negPoints.append(docNames[j])
         negLabels.update(docDicts[docNames[j]])
  posNegMap[docNames[i]] = negPoints     
  negLabelDicts[docNames[i]] = negLabels
  degreeMap[cInstance] = cInstMap
  #print negLabels
#  print docDicts[docNames[i]]
#  print "\n"

maxAngle = max(angles)
# Purpose --- Find least similar / best counter example

# 5th part
thresh5th = maxAngle
thresh4th = maxAngle / 5 * 4
thresh3rd = maxAngle / 5 * 3
thresh2th = maxAngle / 5 * 2
thresh1st = maxAngle / 5 * 1
thresh0th = 0

#print "\n\n"
#print posNegMap
#exit(0)
print 'Objective: You are trying to teach a robot about objects (for example, a banana) found in a house by giving examples and counterexamples.'
print 'Help choose the BEST examples (things that are most similar to the object) or BEST counterexamples (things that are LEAST similar to that object)'
print ""
#print 'Question: Please choose the objects that are LEAST similar/BEST counter-examples to the object:  '


for k,v in degreeMap.items() :
   divChange1 = 0
   divChange2 = 0
   divChange3 = 0
   divChange4 = 0
   divChange5 = 0
   print k, " , ,"
   print ''
   ss = sorted(v.items(), key=lambda x: x[1])
   for itemSS in ss:
        if (itemSS[1] >= thresh0th and divChange1 == 0):
            print '1st Part: '
            divChange1 = 1
        if (itemSS[1] > thresh1st and divChange2 == 0):
            print '2nd Part: '
            divChange2 = 1
        if (itemSS[1] > thresh2th and divChange3 == 0):
            print '3rd Part: '
            divChange3 = 1
        if (itemSS[1] > thresh3rd and divChange4 == 0):
            print '4th Part: '
            divChange4 = 1
        if (itemSS[1] > thresh4th and divChange5 == 0):
            print '5th Part: '
            divChange5 = 1
        print ',,', itemSS[0],',,', itemSS[1]
   print "\n\n"
#print thresh5th ,thresh4th, thresh3rd, thresh2th,thresh1st, thresh0th
exit(0)

model = gensim.models.Word2Vec(docLists, min_count=1)
arThreshWords = ['yellow', 'out', 'block', 'arch', 'building','archshaped', 'arc', 'yellowcolour','brightly', 'toy']
firstWord = arThreshWords[0]
#for word in arThreshWords:
#   if word != firstWord:
#      print word,firstWord,(model.similarity(word, firstWord))

#print model.most_similar(positive=['yellow', 'arch'])
#print model.most_similar_cosmul(positive=['yellow', 'arch'])


firstDoc = topTFIDFWordLists[0]
for i,item in enumerate(docNames):
 if i != 0 :
  tDoc = docLists[i]
#  print firstDoc,tDoc
#  print docNames[0],docNames[i],model.n_similarity(firstDoc,tDoc)

prString = ""
firstDoc = topTFIDFWordLists[0]
for i,item in enumerate(docNames):
 if i != 0 :
  tDoc = docLists[i]
#  print firstDoc,tDoc
#  prString += docNames[0] + "," + docNames[i] + "," + str(model.wmdistance(firstDoc,tDoc)) + "\n"

#print prString

firstDoc = topTFIDFWordLists[0]
print model.similarity('arch','archshaped')
print model.similarity('yellowcolour','yellow')
print model.similarity('yellowcolor','yellow')
print model.similarity('blue','yellow')
print model.similarity('arch','wedge')
print model.similarity('arch','block')
print model.similarity('arch','vegetable')
