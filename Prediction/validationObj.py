import os
import numpy as np

resultFile = "ResultsRandom/Random-Object-ML-ComparisonsIII.csv"
os.system(">"+resultFile)
#for cat in ['ML','ALPool','ALUnc']:
for cat in ['ML']:
  for testNo in [0,1,2,3]:
    cmd = "echo \'Random Object - " + cat + " Set " + str(testNo) + " \' >> " + resultFile
    os.system(cmd)
    fN = "ResultsRandom/ActiveLearningResults-Set" + str(testNo) + "/" + str(cat) + "/ObjectFeaturesIII/NoOfDataPoints/"
    script = "../Prediction/macro-pos5DescrNegDocVecdistractorTest.py"
    category = "object"
    cmd = "python " + script + " " + fN + " " + category + " " + category + " >> " + resultFile
    print cmd
    os.system(cmd)
