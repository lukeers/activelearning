# This script tests and validates the performance of visual classifiers beyond category, 
# by selecting right positive instances from the pool of positive and negative instance object images.
#
# Positive instances --> images of instances of which the visual classifier token is used at least 6 times to describe. 
# Negative instances -- > the intersection of negative sampling and Doc2Vec negative instances of all positive instances
#

#!/usr/bin/env python
import numpy as np
from pandas import DataFrame, read_table
import pandas as pd
import random
import util
from collections import Counter
import json
import sys
import csv
import os
import collections

posInsts = {}
negInsts = {}
objInstances = {}
objNames = {}
classifierProbs = {}

argvs = sys.argv
if len(argvs) == 1:
	exit(0)

fld = str(argvs[1])

argProb = 0.50
tID = ""

if len(argvs) > 2:
	tID =  str(argvs[2])

cID = ""
if len(argvs) > 3:
   cID = str(argvs[3])

#tID = "rgb"
#tID = "shape"
#tID = "object"
posName = "objposInstances.conf"
negName = "objnegInstances.conf"

pos = fld + "/" + tID + "objposInstances.conf"
neg = fld + "/" + tID + "objnegInstances.conf"

meaningfulWords = ["wedge", "cylinder", "square", "yellow","carrot", "tomato", "curved", "archshaped","lime", "blue", "eggplant", "purple","cuboid", "prism", "orange", "plantain", "white", "semicylinder", "banana", "red", "cube", "triangle", "semicircle", "cylindrical", "corn", "triangular", "cucumber", "brinjal", "lemon", "cabbage", "arch", "circle",  "plum", "potato", "rectangular", "green", "eggplant",  "rectangle"]

rgbWords  = ["yellow","blue", "purple","orange", "white", "red", "green"]

shapeWords  = ["wedge", "cylinder", "square",  "curved", "archshaped","cuboid",  "semicylinder",  "cube", "triangle", "semicircle", "cylindrical", "triangular","arch", "circle",  "rectangular",  "rectangle"]

objWords = ["cylinder", "carrot", "tomato", "lime", "cuboid", "prism", "orange", "plantain", "semicylinder", "banana",  "cube", "triangle", "corn", "cucumber", "brinjal", "lemon", "cabbage", "arch",  "plum", "eggplant"]

predfName = fld + "/groundTruthPrediction.csv"
predfileName = fld
def getPosNegInstances():
    resDictFile = []

    with open(pos) as csvfile:
        readFile = csv.DictReader(csvfile)
        for row in readFile:
            resDictFile.append(row)
            color = row['tokens']
            objs = row['object']
            if objs in objInstances.keys():
                posInsts[objs] = color   

    

  	  	  	  	  
  	  	  	  	  
def getTestInstancesAndClassifiers(fName):
#print fName
	global objInstances,objNames,classifierProbs
	head = 0
	objInstances = {}
	objNames = {}
	classifierProbs = {}
	with open(fName) as csvfile:
		readFile = csv.DictReader(csvfile)
		for row in readFile:
			if head == 0:
				temp = row.keys()

				temp.remove('Type')
				temp.remove('Token')
				for inst in temp:
					ar = inst.split("-")
					if ar[1] in objInstances.keys():
						objInstances[ar[1]].append(inst)
					else :
						objInstances[ar[1]] = [inst] 
						objNames[ar[1]] = row[inst]
			else :
                                if cID == "":
					#if row['Token'] in posInsts.keys():
					if row['Token'] in classifierProbs.keys():
						classifierProbs[row['Token']].append(row)
					else:
						classifierProbs[row['Token']] = [row]
				else:
					if row['Type'] == cID:
						if row['Token'] in classifierProbs.keys():
							classifierProbs[row['Token']].append(row)
						else:
							classifierProbs[row['Token']] = [row]
					
			head = head + 1
						
def writePosNeg():
 testObjs = objInstances.keys()
 descObjs = util.getDocsForTest(testObjs)
# descObjs = util.getDocuments()
 objTokens = util.sentenceToWordDicts(descObjs)
 tknsGlobal = set()
 posTokens = {}
 negSampleTokens = {}

 mostImpTokens = {}
 for (key,value) in objTokens.items():
    cValue = Counter(value)
    mostImpTokens[key] = []
    for (k1,v1) in cValue.items():
     if v1 > 10:
        mostImpTokens[key].append(k1)
     if v1 > 10:
       if k1 in meaningfulWords:
          tknsGlobal.add(k1)
          if key in posTokens.keys():
             kk1 = posTokens[key]
             kk1.append(k1)
             posTokens[key] = kk1
          else:
             posTokens[key] = [k1]
 posTokens = collections.OrderedDict(sorted(posTokens.items()))
 f = open(fld + "/" + posName, "w")
 title = "object,tokens\n"
 f.write(title)

 for k,v in posTokens.items():
    ll  = str(k) + ","
    ll += "-".join(v)
    ll += "\n"
    f.write(ll)    
 f.close()
 
 
 
 kWord = ["rgb","shape","object"]
 for wd in kWord:
   f = open(fld + "/" + wd + posName, "w")

   sWords = []
   f.write(title)

   if wd == "rgb":
      sWords = rgbWords
   elif wd == "shape":
      sWords = shapeWords
   elif wd == "object":
      sWords = objWords
   for k,v in posTokens.items():
    vv = []
    for v1 in v:
        if v1 in sWords:
            vv.append(v1)
    if len(vv) > 0:
       ll  = str(k) + ","
       ll += "-".join(vv)
       ll += "\n"
       f.write(ll)

   f.close()
 

def selectCorrectImage(c,testInstances):
    probs = classifierProbs[c]
    argMax = argProb
    selInst = []
    for ik in range(len(probs)):
        
        probInst = probs[ik]
        for inst in testInstances:
            
            if float(probInst[inst]) > argMax:
                #  argMax = float(probs[inst])
                selInst.append(inst)
                
    return list(set(selInst))
			
def getMatchNumbers(relevantInst,selInst,testInstances):
	tNo = float(len(testInstances))
	tP = 0.0
	fN = 0.0
	fP = 0.0   
	tps = list(set(relevantInst).intersection(set(selInst)))
	tP = float(len(tps))
	fP = float(len(selInst) - tP)
	fN = float(len(relevantInst) - tP)
	tN = tNo - tP - fN - fP
	return (tP,fN,fP,tN)
	
def getStats(tP,fN,fP,tN):
	acc = (tP + tN)/(tP + fN + fP + tN)
	if (tP + fP) > 0.0:
		prec = tP / (tP + fP)
	else: 
		prec = 0.0
	if (tP + fN) > 0.0:
		rec = tP / (tP + fN)
	else:
		rec = 0.0
	f1s = 0.0
	if (prec + rec) != 0.0:
		f1s = 2.0 * prec * rec / (prec + rec)
	return (acc,prec,rec,f1s)
		
		
		
resultFolder = fld




noIndArs = []
noImgs = []
noSels = []

#for fNo in np.arange(100,2000,100):
for fNo in np.arange(10,110,10):
    fName1 = predfileName + "/" + str(fNo)
    resultFolder = fName1
    fld = resultFolder
    fName = fName1 + "/groundTruthPrediction.csv"
    predfName = fName
    pos = fName1 + "/" + tID + "objposInstances.conf"
    neg = fName1 + "/" + tID + "objnegInstances.conf"
    resultFileName = resultFolder + str(fNo) + "-" + str(tID) + "_" + str(cID) + "_learnPerformance.csv" 
    perfFile = open(resultFileName,'w') 
    fieldnames = np.array(['Objects','Test Object Images','Ground Truth','Selected by Classifier'])
    fieldnames = np.append(fieldnames,['True Positive','True Negative','False Positive','False Negative'])
    fieldnames = np.append(fieldnames,['Accuracy','Precision','Recall','F1-Score'])

    getTestInstancesAndClassifiers(fName)
    writePosNeg()
    getPosNegInstances()
    testObjs = list(set(posInsts.keys()).intersection(set(objInstances.keys())))
    totalDetect = 0  
    totalImgs = 0
    print fName
    for c in testObjs:
      tkns = posInsts[c].split("-")
      hghst = 0
        
      for tkn in tkns:
            if tkn in classifierProbs.keys():
                sInsts = selectCorrectImage(tkn,objInstances[c])
                if len(sInsts) > hghst:
                    hghst = len(sInsts)
      totalImgs += len(objInstances[c])
      totalDetect += hghst
    print totalDetect,totalImgs
    noIndArs.append(str(fNo))
    noImgs.append(str(totalImgs))
    noSels.append(str(totalDetect))


print ", ".join(noIndArs)
print ", ".join(noImgs)
print ", ".join(noSels)




