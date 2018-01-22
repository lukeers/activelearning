#!/usr/bin/env python
import sys
import numpy as np
from pandas import DataFrame, read_table
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import csv
import random
from sklearn.naive_bayes import GaussianNB

##options - 'yellow' ,'green' ,'blue' ,'red' ,'brown','orange','purple','white'}
cls1 = sys.argv[1]
cls2 = sys.argv
cls2[0:2] = []

print "Testing between ", cls1, " and ", cls2

tobeTested = "purple"
execPath = '/Users/nishapillai/Documents/GitHub/alExec/'
dsPath = execPath + "pNmPx/"

########From DataSet #################
yellowCls = ['arch/arch_1','cuboid/cuboid_1','cylinder/cylinder_1','semicylinder/semicylinder_3','triangle/triangle_2','banana/banana_1','lemon/lemon_1','banana/banana_2','banana/banana_3','lemon/lemon_4','lemon/lemon_3','lemon/lemon_2','cube/cube_1']

greenCls = ['banana/banana_4','cuboid/cuboid_2','cucumber/cucumber_3','cucumber/cucumber_1','arch/arch_3','cucumber/cucumber_2','lime/lime_4','triangle/triangle_3','cube/cube_2','cylinder/cylinder_4','semicylinder/semicylinder_4','lime/lime_3','lime/lime_1','lime/lime_2','cucumber/cucumber_2']

blueCls = ['arch/arch_2','cuboid/cuboid_4','cylinder/cylinder_3','semicylinder/semicylinder_1','triangle/triangle_4','cube/cube_3']

redCls = ['semicylinder/semicylinder_2','cylinder/cylinder_2','cuboid/cuboid_3','arch/arch_4','triangle/triangle_1','cube/cube_4','tomato/tomato_1','tomato/tomato_2','tomato/tomato_3','plum/plum_1','plum/plum_2','plum/plum_3','plum/plum_4','potato/potato_1','potato/potato_2','cabbage/cabbage_4','cabbage/cabbage_2','cabbage/cabbage_3','cabbage/cabbage_1']

brownCls = ['potato/potato_1','potato/potato_2','potato/potato_3','potato/potato_4','corn/corn_1']

blackCls = ['arch/arch_2','arch/arch_3','arch/arch_4','cuboid/cuboid_1','cuboid/cuboid_2','cube/cube_1','cube/cube_2','semicylinder/semicylinder_1','semicylinder/semicylinder_3','semicylinder/semicylinder_4','triangle/triangle_2','triangle/triangle_4','lemon/lemon_4','orange/orange_1','orange/orange_2','orange/orange_3','banana/banana_1','banana/banana_2','banana/banana_3','banana/banana_4','cabbage/cabbage_2','cucumber/cucumber_1','cucumber/cucumber_2','cucumber/cucumber_3','cylinder/cylinder_1','cylinder/cylinder_2','cylinder/cylinder_3','corn/corn_2','eggplant/eggplant_3','eggplant/eggplant_4','plum/plum_1','plum/plum_3','plum/plum_4','potato/potato_1']

orangeCls = ['orange/orange_1','orange/orange_2','orange/orange_3','orange/orange_4','tomato/tomato_4','carrot/carrot_1','carrot/carrot_2','carrot/carrot_4']

purpleClas = ['cabbage/cabbage_1','cabbage/cabbage_2','cabbage/cabbage_3','cabbage/cabbage_4','eggplant/eggplant_1','eggplant/eggplant_2','eggplant/eggplant_3','eggplant/eggplant_4','plum/plum_3','potato/potato_1']


##############From ground truth
yellowCls = ['arch/arch_1','banana/banana_1','banana/banana_2','banana/banana_3','cube/cube_1','cuboid/cuboid_1','cylinder/cylinder_1','lemon/lemon_1','lemon/lemon_2','lemon/lemon_4','lemon/lemon_3','semicylinder/semicylinder_3','triangle/triangle_2']

greenCls = ['arch/arch_3','banana/banana_4','cube/cube_2','cuboid/cuboid_2','cucumber/cucumber_3','cucumber/cucumber_4','cucumber/cucumber_1','cucumber/cucumber_2','cylinder/cylinder_4','semicylinder/semicylinder_4','lime/lime_3','lime/lime_1','lime/lime_2','lime/lime_4','triangle/triangle_3']

blueCls1 = ['arch/arch_2','cuboid/cuboid_4','cylinder/cylinder_3','semicylinder/semicylinder_1','triangle/triangle_4','cube/cube_3']

redCls1 = ['arch/arch_4','cube/cube_4','cuboid/cuboid_3','cylinder/cylinder_2','semicylinder/semicylinder_2','tomato/tomato_1','tomato/tomato_2','tomato/tomato_3','tomato/tomato_4','triangle/triangle_1']

brownCls = ['potato/potato_1','potato/potato_2','potato/potato_3','potato/potato_4']

orangeCls = ['orange/orange_1','orange/orange_2','orange/orange_3','orange/orange_4','carrot/carrot_1','carrot/carrot_2','carrot/carrot_3','carrot/carrot_4']

purpleCls = ['cabbage/cabbage_1','cabbage/cabbage_2','cabbage/cabbage_3','cabbage/cabbage_4','plum/plum_1','plum/plum_2','plum/plum_3','plum/plum_4','eggplant/eggplant_1','eggplant/eggplant_2','eggplant/eggplant_3','eggplant/eggplant_4']

whiteCls = ['corn/corn_1','corn/corn_2','corn/corn_3','corn/corn_4']
#print len(redCls1)
redCls = []
rindices = [0,1,2,3,5,9]

for ind in rindices:
   redCls.append(redCls1[ind])

#print "Total Blues ", len(blueCls1)
blueCls = []
rindices = [3]

for ind in rindices:
   blueCls.append(blueCls1[ind])

gTMap = {'yellow':yellowCls,'green':greenCls,'blue':blueCls,'red':redCls,'brown':brownCls,'orange':orangeCls,'purple':purpleCls,'white':whiteCls}

##############From ToySet
yellowCls = ['arch/arch_1','cube/cube_1','cuboid/cuboid_1','cylinder/cylinder_1','semicylinder/semicylinder_3','triangle/triangle_2']

greenCls = ['arch/arch_3','cube/cube_2','cuboid/cuboid_2','cylinder/cylinder_4','semicylinder/semicylinder_4','triangle/triangle_3']

blueCls1 = ['arch/arch_2','cuboid/cuboid_4','cylinder/cylinder_3','semicylinder/semicylinder_1','triangle/triangle_4','cube/cube_3']

redCls1 = ['arch/arch_4','cube/cube_4','cuboid/cuboid_3','cylinder/cylinder_2','semicylinder/semicylinder_2','triangle/triangle_1']


blueCls = blueCls1
redCls = redCls1

gTMap = {'yellow':yellowCls,'green':greenCls,'blue':blueCls,'red':redCls}

class1 = gTMap[cls1]

class2 = []
if len(cls2) == 0:
 for key,value in gTMap.items():
   if key == tobeTested:
      class1 = value
   else :
      class2.extend(value)
else:
  for value in cls2:
     class2.extend(gTMap[value])

#class1 = yellowCls
#class2 = redCls
#class2.extend(greenCls)
#class2.extend(blueCls)
#class2.extend(brownCls)
#class2.extend(blackCls)
#class2.extend(orangeCls)
#class2.extend(purpleClas)

#class1 = gTMap[cls1]
#class2 = gTMap[cls2]

#print "Classification for ", tobeTested
print len(class1),len(class2)

def getFeatures(instName,kind):
   ar1 = instName.split("/")
   path1 = "/".join([dsPath,instName])
   path  = path1 + "/" + ar1[1] + "_" + kind + ".log"
   featureSet = read_table(path,sep=',',  header=None)
   return featureSet.values

cls1  = "R,G,B,class\n"
cls2 = cls1
for cls in class1:
   for inst in getFeatures(cls,'rgb'):
      cls1 += ",".join([str(int(inst[0] * 255)),str(int(inst[1] * 255)),str(int(inst[2] * 255))])
      cls1 += ",1\n"
      cls2 += ",".join([str(int(inst[0] * 255)),str(int(inst[1] * 255)),str(int(inst[2] * 255))])
      cls2 += ",0\n"

for cls in class2:
   for inst in getFeatures(cls,'rgb'):
      cls2 += ",".join([str(int(inst[0] * 255)),str(int(inst[1] * 255)),str(int(inst[2] * 255))])
      cls2 += ",1\n"
      cls1 += ",".join([str(int(inst[0] * 255)),str(int(inst[1] * 255)),str(int(inst[2] * 255))])
      cls1 += ",0\n"

file = open("class1.txt","w") 
file.write(cls1)
file.close()

file = open("class2.txt","w") 
file.write(cls2)
file.close()

path = 'class1.txt'

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

tNum = int(len(train)/4)
index = [random.randint(0,len(train) - 1) for i in range(tNum)]
#print "Testing indices : ",index

tX = []
X =[]
Y = []
tY = []

for i in index:
   tX.append(train[i])
   tY.append(trainY[i])

for i in range(len(train)):
  if i not in index:
     X.append(train[i])
     Y.append(trainY[i])

print "Total ",len(train)
#print tX
print  tY

logreg = linear_model.LogisticRegression()
#logreg = linear_model.SGDClassifier()
#logreg = GaussianNB()
logreg.fit(X, Y)
predY = logreg.predict(tX)
(prec,recall,f1score,_)  = precision_recall_fscore_support(tY, predY,average='micro')
acc = logreg.score(X, Y)
print "Predicted : ", predY
print "Results :: ",acc,prec,recall,f1score
