#!/usr/bin/env python
import numpy as np
from pandas import DataFrame, read_table
import pandas as pd
import random
from collections import Counter
import json

from sklearn.pipeline import Pipeline
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import csv 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

kinds = np.array(['rgb','shape'])
execPath = '/Users/nishapillai/Documents/GitHub/alExec/'
dsPath = execPath + "nDz/"
fAnnotation = execPath + "groundtruth_annotation.conf"

cGT = {}
sGT = {}
oGT = {}

generalColors = ['yellow','blue','purple','black','isyellow','green','brown','orange','white','red']

generalObjs = ['potatoe','cylinder','square', 'cuboid', 'sphere', 'halfcircle','circle','rectangle','cube','triangle','arch','semicircle','halfcylinder','wedge','block','apple','carrot','tomato','lemon','cherry','lime', 'banana','corn','hemisphere','cucumber','cabbage','ear','potato', 'plantain','eggplant']

generalShapes = ['spherical', 'cylinder', 'square', 'rounded', 'cylindershaped', 'cuboid', 'rectangleshape','arcshape', 'sphere', 'archshaped', 'cubeshaped', 'curved' ,'rectangular', 'triangleshaped', 'halfcircle', 'globular','halfcylindrical', 'circle', 'rectangle', 'circular', 'cube', 'triangle', 'cubic', 'triangular', 'cylindrical','arch','semicircle', 'squareshape', 'arched','curve', 'halfcylinder', 'wedge', 'cylindershape', 'round', 'block', 'cuboidshaped']


class Category:
   __slots__ = ['catNums', 'name']  
   catNums = np.array([], dtype='object')
   def __init__(self, name):
        self.name = name
   def getName(self):
      return self.name
     
   def addCategories(self,*num):
       self.catNums = np.unique(np.append(self.catNums,num))

   def chooseOneInstance(self):
      r = random.randint(0,self.catNums.size - 1)  
      instName = self.name + "/" + self.name + "_" + self.catNums[r]
      return instName

class Instance(Category):
    __slots__ = ['name','catNum','tokens','fS','negs','gT']
    gT = {}
    tokens = np.array([])
    fS = {}
    def __init__(self, name,num):
        self.name = name
        self.catNum = num

    def getName(self):
       return self.name

    def findFeatureValues(self,dsPath):
        instName = self.name
        instName.strip()
        ar1 = instName.split("/")
        path1 = "/".join([dsPath,instName])
        for kind in kinds:
           path  = path1 + "/" + ar1[1] + "_" + kind + ".log"
           featureSet = read_table(path,sep=',',  header=None)
           self.fS[kind] = featureSet

    def getFeatures1(self,kind):
       return self.fS[kind].values

    def getFeatures(self,kind):
        instName = self.name
        instName.strip()
        ar1 = instName.split("/")
        path1 = "/".join([dsPath,instName])
        path  = path1 + "/" + ar1[1] + "_" + kind + ".log"
        featureSet = read_table(path,sep=',',  header=None)
        return featureSet.values

    def addNegatives(self, negs):
       add = lambda x : np.unique(map(str.strip,x))
       self.negs = add(negs)

    def getNegatives(self):
      return self.negs

    def addTokens(self,tkn):
        self.tokens = np.append(self.tokens,tkn)

    def getTokens(self):
       return self.tokens

    def addY(self,dsYs,kind) :
       self.gT.update({kind:dsYs})

    def getY(self,token,kind):
      if token in self.gT[kind]:
         return 1
      return 0
  
    def getY1(self,token,kind):
       if token in list(self.tokens):
          if kind == "rgb":
              if token in list(generalColors):
                 return 1
          elif kind == "shape":
             if token in list(generalShapes):
                 return 1
          else:
             if token in list(generalObjs):
                return 1
       return 0

class Token:
   __slots__ = ['name', 'posInstances', 'negInstances']
   posInstances = np.array([], dtype='object')
   negInstances = np.array([], dtype='object')

   def __init__(self, name):
        self.name = name
   
   def getTokenName(self):
       return self.name

   def extendPositives(self,instName):
      self.posInstances = np.append(self.posInstances,instName)
   
   def getPositives(self): 
      return self.posInstances

   def extendNegatives(self,*instName):
      self.negInstances = np.unique(np.append(self.negInstances,instName))

   def getNegatives(self):
      return self.negInstances

   def clearNegatives(self):
      self.negInstances = np.array([])

   def shuffle(self,a, b, rand_state):
      rand_state.shuffle(a)
      rand_state.shuffle(b)

   def getTrainFiles(self,insts,kind):
      instances = insts.to_dict()
      #print "Pos : ",self.posInstances
      #print "Neg : ",self.negInstances
      features = np.array([])
      negFeatures = np.array([])
      y = np.array([])
      if self.posInstances.shape[0] == 0 or self.negInstances.shape[0] == 0 :
         return (features,y)
      if self.posInstances.shape[0] > 0 :
        features = np.vstack(instances[inst][0].getFeatures(kind) for inst in self.posInstances)
      if self.negInstances.shape[0] > 0:
        negFeatures = np.vstack(instances[inst][0].getFeatures(kind) for inst in self.negInstances if len(inst) > 1)
        if len(features) > len(negFeatures):
          c = int(len(features) / len(negFeatures))
          negFeatures = np.tile(negFeatures,(c,1))
      if self.posInstances.shape[0] > 0 and self.negInstances.shape[0] > 0 :
       if len(negFeatures) > len(features):
          c = int(len(negFeatures) / len(features))
          features = np.tile(features,(c,1))
      y = np.concatenate((np.full(len(features),1),np.full(len(negFeatures),0)))
      if self.negInstances.shape[0] > 0:
        features = np.vstack([features,negFeatures])
      #self.shuffle(features,y, np.random.RandomState(12345))
      return(features,y)


class DataSet:
   __slots__ = ['dsPath', 'annotationFile']
   def __init__(self, path,anFile):
      self.dsPath = path
      self.annotationFile = anFile

   def addNegativeToInstances(self):
      nDf = read_table(self.negDatasetCollection,sep=':',  header=None)
      nDs = nDf.values
      categories = {}
      instances = {}
      for (k1,v1) in nDs:
          instName = k1.strip()
          (cat,inst) = instName.split("/")
          (_,num) = inst.split("_")
          if cat not in categories.keys():
             categories[cat] = Category(cat)
          categories[cat].addCategories(num)
          if instName not in instances.keys():
             instances[instName] = Instance(instName,num)
          instances[instName].addNegatives(v1.split(","))
    
#      print instances['arch/arch_1'].getNegatives()
      instDf = pd.DataFrame(instances,index=[0])
      catDf =  pd.DataFrame(categories,index=[0])
      return (catDf,instDf)

   def findCategoryInstances(self):
      nDf = read_table(self.annotationFile,sep=',',  header=None)
      nDs = nDf.values
      categories = {}
      instances = {}
      for (k1,v1) in nDs:
          instName = k1.strip()
          (cat,inst) = instName.split("/")
          (_,num) = inst.split("_")
          if cat not in categories.keys():
             categories[cat] = Category(cat)
          categories[cat].addCategories(num)
          if instName not in instances.keys():
             instances[instName] = Instance(instName,num)
      instDf = pd.DataFrame(instances,index=[0])
      catDf =  pd.DataFrame(categories,index=[0])
      return (catDf,instDf)


   def splitTestInstances(self,cDf):
      cats = cDf.to_dict()
      negInstances = np.array([])
      for (cat, obj) in cats.items():
         negInstances = np.append(negInstances,obj[0].chooseOneInstance())
      return negInstances

   def getDataSet(self):
      (cDf,nDf) = self.findCategoryInstances()
      tests = self.splitTestInstances(cDf)
      instances = nDf.to_dict()
      df = read_table(self.annotationFile, sep=',',  header=None)  
      tokenDf = {}
      for column in df.values:
        ds = column[0]
        dsTokens = column[1].split(" ")
        dsTokens = list(filter(None, dsTokens))
        instances[ds][0].addTokens(dsTokens)

        if ds not in tests:
         for annotation in dsTokens:
             if annotation not in tokenDf.keys():
                 tokenDf[annotation] = Token(annotation)
             tokenDf[annotation].extendPositives(ds) 
      tks = pd.DataFrame(tokenDf,index=[0])
      return (nDf,tks,tests)

   def getAllFeatures(self,nDf):
      instances = nDf.to_dict()
      for inst in instances.keys():
         objInst = instances[inst][0]
         objInst.findFeatureValues(dsPath)

def getGroundTruth(nInst,token,kind):
   gt = ""
   if kind == "rgb":
      gt = cGT
   elif kind == "shape":
      gt = sGT
   else:
      gt = oGT
   if token in gt[nInst]:
      return 1
   return 0
    


              
def getTestFiles(insts,kind,tests,token):
   instances = insts.to_dict()
   features = []
   y = []
   #print "Test ",
   for nInst in tests:
#      y1 = instances[nInst][0].getY(token,kind)
      y1 = getGroundTruth(nInst,token,kind)
      fs  = instances[nInst][0].getFeatures(kind)
      #print nInst,Counter(instances[nInst][0].getTokens())
      features.append(list(fs))
      y.append(list(np.full(len(fs),y1)))
   return(features,y)

def  findScoresManual(ttY,predY):
   tP = 0
   fP = 0
   fN = 0
   for j in range(len(ttY)):
      r = ttY[j]
      s = predY[j]
      if(r == 1) :
         if s == 1:
           tP += 1
         else:
           fN += 1
      elif(s == 1):
        fP += 1
   prec = 1.0
   rec = 1.0
   if(tP + fP) != 0 :
     prec = float(tP / (tP + fP))
   if (tP + fN) != 0 :
     rec = float(tP / (tP + fN))

   f1 = 0.0
   if(prec + rec) != 0:
     f1 = float(2 * prec * rec / (prec + rec))
   return (prec,rec,f1)
    

def findTrainTestFeatures(insts,tkns,tests):
  tokenDict = tkns.to_dict()
#  for token in ['blue']:
  for token in tokenDict.keys():
     objTkn = tokenDict[token][0]
     for kind in kinds:
#     for kind in ['rgb']: 
        (features,y) = objTkn.getTrainFiles(insts,kind)
        (testFeatures,testY) = getTestFiles(insts,kind,tests,token)
        if len(features) == 0 :
            continue;
        yield (token,kind,features,y,testFeatures,testY)

def Experiments(X,Y):
#   logreg = linear_model.LogisticRegression()
#   logreg = linear_model.SGDClassifier(loss='log',penalty='elasticnet',l1_ratio=0.05,n_iter=1000)
#   logreg = linear_model.SGDClassifier(loss='log')
#   logreg.fit(X, Y)
#   print "done - normal"
#   polynomial_features = PolynomialFeatures(degree=1,include_bias=False)
#   sgdK = linear_model.LogisticRegression(C=10**-5,random_state=0)
#   pipeline = Pipeline([("polynomial_features", polynomial_features),
#                         ("logistic", sgdK)])
#   pipeline.fit(X,Y)
#   print "done - 1"
#   polynomial_features1 = PolynomialFeatures(degree=2,include_bias=False)
#   sgdK1 = linear_model.LogisticRegression(C=10**-5,random_state=0)
#   pipeline2 = Pipeline([("polynomial_features", polynomial_features1),
#                         ("logistic", sgdK1)])
#   pipeline2.fit(X,Y)
   print "done-2"
   polynomial_features = PolynomialFeatures(degree=3,include_bias=False)
   sgdK = linear_model.LogisticRegression(C=10**-5,random_state=0)
   pipeline3 = Pipeline([("polynomial_features", polynomial_features),
                         ("logistic", sgdK)])
#   pipeline3.fit(X,Y)
#   print "done - 3"
   polynomial_features = PolynomialFeatures(degree=1,include_bias=False)
   sgdK = linear_model.LogisticRegression(C=10**5,random_state=0)
   pipeline1_2 = Pipeline([("polynomial_features", polynomial_features),
                         ("logistic", sgdK)])
   pipeline1_2.fit(X,Y)
   print "done - 1_2"
   polynomial_features = PolynomialFeatures(degree=2,include_bias=False)
   sgdK = linear_model.LogisticRegression(C=10**5,random_state=0)
   pipeline2_2 = Pipeline([("polynomial_features", polynomial_features),
                         ("logistic", sgdK)])



def callML(insts,tkns,tests):
  confFile = open('groundTruthConfMatrix.csv','w')
  fldNames = np.array(['Token','Type'])
  fldNames = np.append(fldNames,tests)
  confWriter = csv.DictWriter(confFile, fieldnames=fldNames)
  confWriter.writeheader()

  csvFile = open('groundTruthResults.csv', 'w')
  fieldnames = np.array(['Token','Type'])
  fieldnames = np.append(fieldnames,['Accuracy','Precision','Recall','F1-Score','Results'])
  fieldnames = np.append(fieldnames,tests)
  writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
  writer.writeheader()
  featureSet = read_table(fAnnotation,sep=',',  header=None)
  featureSet = featureSet.values
  fSet = dict(zip(featureSet[:,0],featureSet[:,1]))
  confD = {}
  for tt in tests:
     confD[tt] = str(fSet[tt])
  confWriter.writerow(confD)
  pNum = 4
  for (token,kind,X,Y,tX,tY) in findTrainTestFeatures(insts,tkns,tests):
   print "Token : " + token + ", Kind : " + kind
   polynomial_features = PolynomialFeatures(degree=2,include_bias=False)
   sgdK = linear_model.LogisticRegression(C=10**5,random_state=0)
   pipeline2_2 = Pipeline([("polynomial_features", polynomial_features),
                         ("logistic", sgdK)])
   pipeline2_2.fit(X,Y)
   dict1 = {}
   ttX = []
   ttY = []
   tAcc = []
   tPrec = []
   tRec = []
   confDict = {'Token' : token,'Type' : kind}
   for ii in range(len(tX)) :
      testX = tX[ii]
      testY = tY[ii]
      ttX.extend(testX)
      ttY.extend(testY)
      tt = tests[ii]
#      predY = logreg.predict(testX)
#      (prec,recall,f1score,_)  = precision_recall_fscore_support(testY, predY,average='binary')
#      acc = logreg.score(testX, testY)
#      res = "Acc : " + str(round(acc,pNum)) + ", Prec : " + str(round(prec,pNum)) + ",Recall : " + str(round(recall,pNum)) + ", F1-Score : " + str(round(f1score,pNum))
#      dict1[tt] = res
#      probl = logreg.predict_proba(testX)
#      tProbs = probl[:,1]
#      print token, kind, " -- ",tt,"'" + str(fSet[tt]) + "' -" , float(sum(tProbs)/len(tProbs)),
      predY = pipeline2_2.predict(testX)
#      (prec,recall,f1score,_)  = precision_recall_fscore_support(testY, predY,average='binary')
      (prec,recall,f1score) = findScoresManual(testY, predY)
      acc = pipeline2_2.score(testX, testY)
      res = "Acc : " + str(round(acc,pNum)) + ", Prec : " + str(round(prec,pNum)) + ",Recall : " + str(round(recall,pNum)) + ", F1-Score : " + str(round(f1score,pNum))
#      print res
      tAcc.append(acc)
      tPrec.append(prec)
      tRec.append(recall)
      dict1[tt] = res
      probK = pipeline2_2.predict_proba(testX)
      tProbs = probK[:,1]
#      print float(sum(tProbs)/len(tProbs)),
      tConf = float(sum(tProbs)/len(tProbs))
#      confDict[tt] = str(tConf) + " [" + ",".join(str(tProbs)) + "] "
      confDict[tt] = str(tConf)
#      print "\n"
    
   confWriter.writerow(confDict)
#   predY = pipeline2_2.predict(ttX)
   #(prec,recall,f1score,_)  = precision_recall_fscore_support(ttY, predY,average='binary')
#   (prec,recall,f1score) = findScoresManual(ttY,predY)
#   acc = pipeline2_2.score(ttX, ttY)   
   acc = float(sum(tAcc)/len(tAcc))
   prec = float(sum(tPrec)/len(tPrec))
   recall = float(sum(tRec)/len(tRec))
   f1score = float(2.0 * prec * recall / (prec + recall))
   print "Total ", acc,prec,recall,f1score
   probl = pipeline2_2.predict_proba(ttX)
   dict2 = {'Token' : token,'Type' : kind,'Accuracy' : round(acc,pNum) ,'Precision' : round(prec,pNum) ,'Recall' : round(recall,pNum),'F1-Score' : round(f1score,pNum)}
   dict2.update(dict1)
#   writer.writerow({'Token' : token,'Type' : kind,'Accuracy' : round(acc,pNum) ,'Precision' : round(prec,pNum) ,'Recall' : round(recall,pNum),'F1-Score' : round(f1score,pNum),'Results': ",".join(str(predY))})
   writer.writerow(dict2)
  csvFile.close()
  confFile.close()

def generateNegativeTrainingFiles(nDf,tkns,tests):
   instances = nDf.to_dict()
   tokenDict = tkns.to_dict()
#   for token in ['yellow']:
   for token in tokenDict.keys():
     objTkn = tokenDict[token][0]
     objTkn.clearNegatives()
     tknAr = []
     for inst in instances.keys():
      objInst = instances[inst][0]
      if (inst not in tests) and (token not in objInst.getTokens()):
       tknAr.append(objInst.getName())
     objTkn.extendNegatives(tknAr)
     
def getAllGroundTruths(nDf,cGTFile,sGTFile,oGTFile):
   instances = nDf.to_dict()
   gtFile = ""
   gt = ""
   for kind in kinds:
    if kind == "rgb":
      gtFile = cGTFile
      gt = cGT
    elif kind == "shape":
      gtFile = sGTFile
      gt = sGT
    else:
      gtFile = oGTFile
      gt = oGT
    df = read_table(gtFile, sep=',',  header=None)
    for column in df.values:
      ds = column[0]
      dsYs = column[1].split(" ")
      dsYs = list(filter(None, dsYs))
      gt[ds] = dsYs

if __name__== "__main__":

  anFile =  execPath + "groundtruth_annotation.conf"
  cGTFile = execPath + "color_groundtruth_annotation.conf"
  sGTFile = execPath + "shape_groundtruth_annotation.conf"
  oGTFile = execPath + "object_groundtruth_annotation.conf"

  ds = DataSet(dsPath,anFile)
  (insts,tokens,tests) = ds.getDataSet()
  generateNegativeTrainingFiles(insts,tokens,tests)

  getAllGroundTruths(insts,cGTFile,sGTFile,oGTFile)  

  ds.getAllFeatures(insts)

  callML(insts,tokens,tests)
  print "Hi Welcome to AL !!"

#  instances = insts.to_dict()
#  print Counter(instances['arch/arch_1'][0].getTokens())
#  tks = tokens.to_dict()
#  print Counter(tks['block'][0].getPositives())
