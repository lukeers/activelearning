#!/usr/bin/env python
import numpy as np
from pandas import DataFrame, read_table
import pandas as pd

class Category:
   
   catNums = np.array([], dtype='object')

   def __init__(self, name):
        self.name = name
        
   def addCategories(self,*num):
       self.catNums = np.append(self.catNums,num)

class Instance(Category):
   
    def __init__(self, name,num):
        self.name = name
        self.catNum = num

    def findFeatureValues(self,dsPath):
        instName = self.name
        instName.trim()
        path = dsPath + "/" + instName + "/"
        for kind in ['color','shape','object']:
           if kind == 'color':
              path += "/rgb.log"
           elif kind == 'shape':
              path += "/shape.log"
           else:
              path += "/object.log"
           self.path = path
           featureSet = read_table(path,sep=',',  header=None)
           self.fS[kind] = np.array(featureSet)

    def getFeatures(self,kind):
       return self.fS[kind] 
    

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

   def getDataSet(self):
      nDf = read_table(self.negDatasetCollection,sep=':',  header=None)
      nDs = nDf.values
      negInstances = {}
      for (k1,v1) in nDs:
          negInstances[k1] = np.array(v1.split(","))
          np.delete(negInstances[k1],[],0)

      df = read_table(self.annotationFile, sep=',',  header=None)  
#      tokenDf = pd.DataFrame(index=None, columns=['Token','Object'])
      tokenDf = {}
      for column in df.values:
         for annotation in column[1].split(" "):
            if annotation is not "":
              if annotation not in tokenDf.keys():
                 tokenDf[annotation] = Token(annotation)
             #    tokenDf.loc[len(tokenDf)] = [[annotation,tk]]
             #    tokenDf = tokenDf.append([{annotation:tk}])
              tokenDf[annotation].extendPositives(column[0]) 
              tokenDf[annotation].extendNegatives(negInstances[column[0]])
      #print tokenDf
      #print "\n\n"
      #print negInstances['arch/arch_1']
      #print tokenDf['red'].getNegatives()

if __name__== "__main__":
  t = Token("red")
  print t.getTokenName()
  t.extendPositives('arch/arch1')
  print t.getPositives()
  t.extendNegatives('arch/arch1')
  print t.getNegatives()
  t.extendNegatives('arch/arch2')
  print t.getNegatives()
  t.extendNegatives('arch/arch2','ss')
  print t.getNegatives()

  dsPath = "/home/npillai1/AL/pyal/images"
  anFile = "/home/npillai1/AL/ConfFile/3k_Thresh_fulldataset.conf"
  negFile = "/home/npillai1/AL/ConfFile/negLabels.log"
  ds = DataSet(dsPath,anFile,negFile)
  ds.getDataSet()
  print "Hi Welcome to AL !!"

