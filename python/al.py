#!/usr/bin/env python
import numpy as np
from pandas import DataFrame, read_table
import pandas as pd
import random
from collections import Counter

from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import csv 

kinds = np.array(['rgb','shape','object'])
execPath = '/Users/nishapillai/Documents/GitHub/alExec/'
dsPath = execPath + "pNmPx/"
fAnnotation = execPath + "annotation.conf"


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
    __slots__ = ['name','catNum','tokens','fS','negs']

    tokens = np.array([])
    fS = {}
    def __init__(self, name,num):
        self.name = name
        self.catNum = num

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

    def getY(self,token,kind):
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

   def shuffle(self,a, b, rand_state):
      rand_state.shuffle(a)
      rand_state.shuffle(b)

   def getTrainFiles(self,insts,kind):
      instances = insts.to_dict()
      features = np.vstack(instances[inst][0].getFeatures(kind) for inst in self.posInstances)
      negFeatures = np.vstack(instances[inst][0].getFeatures(kind) for inst in self.negInstances if len(inst) > 1)
      if len(features) > len(negFeatures):
          c = int(len(features) / len(negFeatures))
          negFeatures = np.tile(negFeatures,(c,1))
      if len(negFeatures) > len(features):
          c = int(len(negFeatures) / len(features))
          features = np.tile(features,(c,1))
      y = np.concatenate((np.full(len(features),1),np.full(len(negFeatures),0)))
      features = np.vstack([features,negFeatures])
      features = features * 255
      self.shuffle(features,y, np.random.RandomState(12345))
      return(features,y)


class DataSet:
   __slots__ = ['dsPath', 'annotationFile', 'negDatasetCollection']
   def __init__(self, path,anFile,negFile):
      self.dsPath = path
      self.annotationFile = anFile
      self.negDatasetCollection = negFile

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

   def splitTestInstances(self,cDf):
      cats = cDf.to_dict()
      negInstances = np.array([])
      for (cat, obj) in cats.items():
         negInstances = np.append(negInstances,obj[0].chooseOneInstance())
      return negInstances

   def getDataSet(self):
      (cDf,nDf) = self.addNegativeToInstances()
      tests = self.splitTestInstances(cDf)
#      print negs
      instances = nDf.to_dict()
#      print instances['arch/arch_1'][0].getName()
      df = read_table(self.annotationFile, sep=',',  header=None)  
#      tokenDf = pd.DataFrame(index=None, columns=['Token','Object'])
      tokenDf = {}
      for column in df.values:
        ds = column[0]
        dsTokens = column[1].split(" ")
        dsTokens = list(filter(None, dsTokens))
        instances[ds][0].addTokens(dsTokens)

        if ds not in tests:
         negatives = instances[ds][0].getNegatives()
         negatives = [xx for xx in negatives if xx not in tests]
         for annotation in dsTokens:
             if annotation not in tokenDf.keys():
                 tokenDf[annotation] = Token(annotation)
             #    tokenDf.loc[len(tokenDf)] = [[annotation,tk]]
             #    tokenDf = tokenDf.append([{annotation:tk}])
             tokenDf[annotation].extendPositives(ds) 
             tokenDf[annotation].extendNegatives(negatives)
      tks = pd.DataFrame(tokenDf,index=[0])
      return (nDf,tks,tests)

   def getAllFeatures(self,nDf):
      instances = nDf.to_dict()
      for inst in instances.keys():
         objInst = instances[inst][0]
         objInst.findFeatureValues(dsPath)
              
def getTestFiles(insts,kind,tests,token):
   instances = insts.to_dict()
   features = np.array([])
   y = np.array([])
   for nInst in tests:
      y1 = instances[nInst][0].getY(token,kind)

      #print nInst,Counter(instances[nInst][0].getTokens())
      if features.size == 0 :
        features = instances[nInst][0].getFeatures(kind)
        y = np.full(len(features),y1)
      else :
        fs  = instances[nInst][0].getFeatures(kind)
        features = np.vstack([features, fs])
        y = np.append(y,np.full(len(fs),y1))
   features = features * 255
   return(features,y)
    

def findTrainTestFeatures(insts,tkns,tests):
  tokenDict = tkns.to_dict()
  for token in ['yellow']:
#  for token in tokenDict.keys():
     objTkn = tokenDict[token][0]
#     for kind in kinds:
     for kind in ['rgb']: 
        (features,y) = objTkn.getTrainFiles(insts,kind)
        (testFeatures,testY) = getTestFiles(insts,kind,tests,token)
        yield (token,kind,features,y,testFeatures,testY)

def callML(insts,tkns,tests):
  csvFile = open('results.csv', 'w')
  fieldnames = np.array(['Token','Type'])
  fieldnames = np.append(fieldnames,['Accuracy','Precision','Recall','F1-Square','Results'])
  fieldnames = np.append(fieldnames,tests)
  writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
  writer.writeheader()
  featureSet = read_table(fAnnotation,sep=',',  header=None)
  featureSet = featureSet.values
  fSet = dict(zip(featureSet[:,0],featureSet[:,1]))
  pNum = 4
  for (token,kind,X,Y,testX,testY) in findTrainTestFeatures(insts,tkns,tests):
   print "Token : " + token + ", Kind : " + kind
   logreg = linear_model.Perceptron()
   logreg.fit(X, Y)
   predY = logreg.predict(testX)
   (prec,recall,f1square,_)  = precision_recall_fscore_support(testY, predY,average='binary')
   acc = logreg.score(testX, testY)   
   print zip(range(len(tests)),tests)
   print zip(range(len(tests)),map(lambda x : fSet[x.split("/")[1]],tests))
   print testY
   print predY
   print acc,prec,recall,f1square
   #print testX
   #print logreg.decision_function(testX)
   exit(0)
   for ls in ['hinge','log','modified_huber','squared_hinge','perceptron']:
  
    for pen in ['l1','l2','elasticnet']:
     for lrate in ['constant','optimal','invscaling']:
      for et in np.arange(0.5,10.0,0.5):
         for iter in np.arange(10,10000,50):
            print ls,pen,lrate,et,iter
            logreg = linear_model.SGDClassifier(loss=ls,penalty=pen,n_iter=iter,learning_rate=lrate,eta0=et,class_weight='balanced')
#            logreg = linear_model.SGDClassifier(penalty='l2',dual=False,tol=0.0001,C=c,class_weight=None,random_state=None,solver=sl,max_iter=iter)
            logreg.fit(X, Y)
            predY = logreg.predict(testX)
            (prec,recall,f1square,_)  = precision_recall_fscore_support(testY, predY,average='binary')
            acc = logreg.score(testX, testY)
            print zip(range(len(tests)),tests)
            print zip(range(len(tests)),map(lambda x : fSet[x.split("/")[1]],tests))
            print testY
            print predY
            print acc,prec,recall,f1square
            print "\n\n"
     
    exit(0)
    print zip(range(len(tests)),tests)
    print zip(range(len(tests)),map(lambda x : fSet[x.split("/")[1]],tests))  
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y)
    predY = logreg.predict(testX)  
    (prec,recall,f1square,_)  = precision_recall_fscore_support(testY, predY,average='binary')
    acc = logreg.score(testX, testY)
    #print list(logreg.decision_function(testX))
    print testY
    print predY
    print acc,prec,recall,f1square
   
    exit(0)
    if kind == "rgb":
        writer.writerow({'Type':'Ground Truth','Results':",".join(str(testY))})
    writer.writerow({'Token' : token,'Type' : kind,'Accuracy' : round(acc,pNum) ,'Precision' : round(prec,pNum) ,'Recall' : round(recall,pNum),'F1-Square' : round(f1square,pNum),'Results': ",".join(str(predY))})
  csvFile.close()

if __name__== "__main__":

  anFile = execPath + "3k_Thresh_fulldataset.conf"
  negFile = execPath + "negLabels.log"
  
  ds = DataSet(dsPath,anFile,negFile)
  (insts,tokens,tests) = ds.getDataSet()
  ds.getAllFeatures(insts)
  callML(insts,tokens,tests)
  print "Hi Welcome to AL !!"

#  instances = insts.to_dict()
#  print Counter(instances['arch/arch_1'][0].getTokens())
#  tks = tokens.to_dict()
#  print Counter(tks['block'][0].getPositives())
