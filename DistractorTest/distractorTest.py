#!/usr/bin/env python
import numpy as np
from pandas import DataFrame, read_table
import pandas as pd
import random
from collections import Counter
import json
import sys
import csv
import os

posInsts = {}
negInsts = {}
objInstances = {}
objNames = {}
classifierProbs = {}

argvs = sys.argv
if len(argvs) == 1:
	exit(0)

fld = str(argvs[1])

argProb = 0.51
tID = ""

if len(argvs) > 2:
	tID =  str(argvs[2])
	if tID == "rgb":
		argProb = 0.50
#tID = "rgb"
#tID = "shape"
#tID = "object"
pos = tID + "posInstances.conf"
neg = tID + "negInstances.conf"


predfName = fld + "/groundTruthPrediction.csv"
def getPosNegInstances():
	resDictFile = []
	with open(pos) as csvfile:
		readFile = csv.DictReader(csvfile)
		for row in readFile:
			resDictFile.append(row)
			color = row['token']
			objs = row['objects']
			insts = objs.split("-")
			newInsts = []
			for instance in insts:
				if instance in objInstances.keys():
					newInsts.append(instance)
					posInsts[color] = newInsts
					
	resDictFile = []
	with open(neg) as csvfile:
  	  readFile = csv.DictReader(csvfile)
  	  for row in readFile:
  	  	  resDictFile.append(row)
  	  	  color = row['token']
  	  	  objs = row['objects']
  	  	  insts = objs.split("-")
  	  	  newInsts = []
  	  	  for instance in insts:
  	  	  	  if instance in objInstances.keys():
  	  	  	  	  newInsts.append(instance)
  	  	  	  	  negInsts[color] = newInsts
  	  	  	  	  
  	  	  	  	  
def getTestInstancesAndClassifiers(fName):
	print fName
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
                                if tID == "":
					#if row['Token'] in posInsts.keys():
					if row['Token'] in classifierProbs.keys():
						classifierProbs[row['Token']].append(row)
					else:
						classifierProbs[row['Token']] = [row]
				else:
					if row['Type'] == tID:
						if row['Token'] in classifierProbs.keys():
							classifierProbs[row['Token']].append(row)
						else:
							classifierProbs[row['Token']] = [row]
					
			head = head + 1
						
						
def getTestImages(testPosInsts,testNegInsts,posNo,negNo):
	posId = []
	negId = []
	testInstances = {}
	relevantInst = []
	if (len(testPosInsts) > 0) and (len(testNegInsts) > 0):
		posId = [(i + 1) %  len(testPosInsts) for i in range(posNo)] 
		negId = [(inn + 1) % len(testNegInsts) for inn in range(negNo)]
		
		for id in posId:
			tmp = testPosInsts[id]
			imgs = objInstances[tmp]
			p1 = random.sample(range(len(imgs)), k=1)
			testInstances[imgs[p1[0]]] = tmp + " (" + objNames[tmp] + ")"
			relevantInst.append(imgs[p1[0]])
      	  
		for id in negId:
			tmp = testNegInsts[id]
			imgs = objInstances[tmp]
			p1 = random.sample(range(len(imgs)), k=1)
			testInstances[imgs[p1[0]]] = tmp + " (" + objNames[tmp] + ")"
	return (relevantInst,testInstances)
      	  
def selectCorrectImage(c,testInstances):
	probs = classifierProbs[c]
	argMax = argProb
	selInst = []
	for ik in range(len(probs)):
		probInst = probs[ik]
		for inst in testInstances.keys():
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
		
		
		
#resultFolder = "GroundedLanguageAcquisitionResults-II/ML-Object/Overall/"
#os.system("rm -rf " + resultFolder)
#os.system("mkdir -p " + resultFolder)

resultFolder = fld

resFileName = resultFolder + tID + "overall-learnPerformance.csv"
perfFile1 = open(resFileName,'w')
fieldnames = np.array(['Test Batch'])
fieldnames = np.append(fieldnames,['True Positive','True Negative','False Positive','False Negative'])
fieldnames = np.append(fieldnames,['Accuracy','Precision','Recall','F1-Score'])
writer1 = csv.DictWriter(perfFile1, fieldnames=fieldnames)
writer1.writeheader()

otP = 0.0
ofN = 0.0
ofP = 0.0
otN = 0.0

for fNo in range(1):
	ttP = 0.0
	tfN = 0.0
	tfP = 0.0
	ttN = 0.0
	# fName = "RGB/Results/" + str(fNo) + "/ML/3500/groundTruthPrediction.csv"
        fName = predfName
	resultFileName = resultFolder + str(fNo) + "-" + str(tID) + "-learnPerformance.csv" 
	perfFile = open(resultFileName,'w') 
	fieldnames = np.array(['Classifier','Test Object Images','Ground Truth','Selected by Classifier'])
	fieldnames = np.append(fieldnames,['True Positive','True Negative','False Positive','False Negative'])
	fieldnames = np.append(fieldnames,['Accuracy','Precision','Recall','F1-Score'])
	writer = csv.DictWriter(perfFile, fieldnames=fieldnames)
	writer.writeheader()
	
	getTestInstancesAndClassifiers(fName)
	getPosNegInstances()
	testTokens = list(set(posInsts.keys()).intersection(set(negInsts.keys())))
	testTokens = list(set(testTokens).intersection(set(classifierProbs.keys())))
	for c in testTokens:
  	  dictRes = {'Classifier' : " "}
  	  writer.writerow(dictRes)
  	  testPosInsts = list(set(posInsts[c]).intersection(set(objInstances.keys())))
  	  testNegInsts = list(set(negInsts[c]).intersection(set(objInstances.keys())))
  	  if len(testPosInsts) > 0 and len(testNegInsts) > 0:
  	  	  for tms in range(10):       
  	  	  	  posNo = random.sample(range(3), k=1)  
  	  	  	  negNo = random.sample(range(6), k=1) 
  	  
  	  	  	  (relevantInst,testInstances) = getTestImages(testPosInsts,testNegInsts,posNo[0] + 1, negNo[0] + 1)
  	  	  	  selInst = selectCorrectImage(c,testInstances)
  	  	  	
  	  	  	  (tP,fN,fP,tN) = getMatchNumbers(relevantInst,selInst,testInstances)
  	  	  	  ttP = ttP + tP
  	  	  	  tfN = tfN + fN
  	  	  	  tfP = tfP + fP
  	  	  	  ttN = ttN + tN
  	  	  	  (acc,prec,rec,f1s) = getStats(tP,fN,fP,tN)
  	  	  	  tmpObj = ""
  	  	  	  for v in testInstances.values():
  	  	  	  	  tmpObj += str(v) + "      "
  	  	  	  relInsts = ""	  	  	
  	  	  	  for ik in relevantInst:	
  	  	  	  	  relInsts += str(testInstances[ik]) + " "
  	  	  	  selInsts = ""
  	  	 
  	  	  	  for ik in selInst:
  	  	  	  	  selInsts += str(testInstances[ik]) + " "
  	  	  	  dictRes = {'Classifier' : str(c),'Test Object Images' : tmpObj, 'Ground Truth' : str(relInsts)}
  	  	  	  dictRes.update({'Selected by Classifier' : str(selInsts), 'Accuracy' : str(acc),'Precision' : str(prec) ,'Recall' : str(rec),'F1-Score' : str(f1s)})
			  dictRes.update({'True Positive' : str(tP),'True Negative' : str(tN) ,'False Positive' : str(fP),'False Negative' : str(fN)})
			 
			  writer.writerow(dictRes)
			    
			  (acc,prec,rec,f1s) = getStats(ttP,tfN,tfP,ttN)
			  dictRes = {'Test Batch' : str(fNo)}
			  dictRes.update({'Accuracy' : str(acc),'Precision' : str(prec) ,'Recall' : str(rec),'F1-Score' : str(f1s)})
			  dictRes.update({'True Positive' : str(ttP),'True Negative' : str(ttN) ,'False Positive' : str(tfP),'False Negative' : str(tfN)})
			  writer1.writerow(dictRes)
			  otP = ttP + otP
			  ofN = tfN + ofN
			  ofP = tfP + ofP
			  otN = ttN + otN
	perfFile.close()   	  	  	  
dictRes = {'Test Batch' : " "}
writer1.writerow(dictRes)
(acc,prec,rec,f1s) = getStats(otP,ofN,ofP,otN)
dictRes = {'Test Batch' : 'Total'}
dictRes.update({'Accuracy' : str(acc),'Precision' : str(prec) ,'Recall' : str(rec),'F1-Score' : str(f1s)})
dictRes.update({'True Positive' : str(otP),'True Negative' : str(otN) ,'False Positive' : str(ofP),'False Negative' : str(ofN)})
writer1.writerow(dictRes)

dRes = {'Test Batch' : 'Total'}
dRes.update({'Accuracy' : str(acc),'Precision' : str(prec) ,'Recall' : str(rec),'F1-Score' : str(f1s)})

print dRes
perfFile1.close()
