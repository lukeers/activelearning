#!/usr/bin/env python
import numpy as np
from pandas import DataFrame, read_table
import pandas as pd
import random
from collections import Counter

class Category:
   
   catNums = np.array([], dtype='object')
   kinds = np.array(['color','shape','object'])
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

    tokens = np.array([])

    def __init__(self, name,num):
        self.name = name
        self.catNum = num

    def findFeatureValues(self,dsPath):
        instName = self.name
        instName.trim()
        path = dsPath + "/" + instName + "/"
        for kind in self.kinds:
           path  += "/" + kind + ".log"
           featureSet = read_table(path,sep=',',  header=None)
           self.fS[kind] = np.array(featureSet)

    def getFeatures(self,kind):
       return self.fS[kind] 
    
    def addNegatives(self, negs):
       add = lambda x : np.unique(map(str.strip,x))
       self.negs = add(negs)

    def getNegatives(self):
      return self.negs

    def addTokens(self,tkn):
        self.tokens = np.append(self.tokens,tkn)

    def getTokens(self):
       return self.tokens

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
   

class DataSet:
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
      negs = self.splitTestInstances(cDf)
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

        if ds not in negs:
         negatives = instances[ds][0].getNegatives()
         negatives = [xx for xx in negatives if xx not in negs]
         for annotation in dsTokens:
             if annotation not in tokenDf.keys():
                 tokenDf[annotation] = Token(annotation)
             #    tokenDf.loc[len(tokenDf)] = [[annotation,tk]]
             #    tokenDf = tokenDf.append([{annotation:tk}])
             tokenDf[annotation].extendPositives(ds) 
             tokenDf[annotation].extendNegatives(negatives)
      tks = pd.DataFrame(tokenDf,index=[0])
      return (nDf,tks,negs)
 
if __name__== "__main__":

  dsPath = "/home/npillai1/AL/pyal/images"
  anFile = "/home/npillai1/AL/ConfFile/3k_Thresh_fulldataset.conf"
  negFile = "/home/npillai1/AL/ConfFile/negLabels.log"
  ds = DataSet(dsPath,anFile,negFile)
  (insts,tokens,negs) = ds.getDataSet()
  print "Hi Welcome to AL !!"

#  instances = insts.to_dict()
#  print Counter(instances['arch/arch_1'][0].getTokens())
#  tks = tokens.to_dict()
#  print Counter(tks['block'][0].getPositives())
