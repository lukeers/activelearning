import numpy as np
from pandas import DataFrame, read_table
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import csv
import random

path = 'dataset/yellow.csv'
featureSet = read_table(path,sep=',',  header=None)
fs = featureSet.values
arTrain = []
for arI in fs[1:len(fs),0:3]:
  arTrain.append([int(j) for j in arI])
train= np.array(arTrain)
arY = []
for j in fs[1:len(fs),3]:
  arY.append(int(j))
trainY = np.array(arY)

index = [random.randint(0,19) for i in range(4)]
print "Testing indices : ",index

X = [train[i]  for i in range(len(train)) if i not in index]
tX = [train[i]  for i in range(len(train)) if i in index]
Y = [trainY[i] for i in range(len(trainY)) if i not in index]
tY = [trainY[i] for i in range(len(trainY)) if i in index]
print X, Y
print tX, tY

logreg = linear_model.LogisticRegression()
logreg.fit(X, Y)
predY = logreg.predict(tX)
(prec,recall,f1square,_)  = precision_recall_fscore_support(tY, predY,average='binary')
acc = logreg.score(X, Y)
print "Predicted : ", predY
print acc,prec,recall,f1square
