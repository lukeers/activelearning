import os
import numpy as np

for testNo in [0,1,2,3]:
# for kWord in ['ML','ALPool','ALUnc']:
 for kWord in ['ML']:
#fN = "ActiveLearningResults-Set3/ALUncertainity/ShapeFeaturesI"
# fN = "ActiveLearningResults-Set" + str(testNo) + "/ALUncertainity/ShapeFeaturesI"
# fN = "ActiveLearningResults-Set" + str(testNo) + "/ALPool/ShapeFeaturesII"
# fN = "ActiveLearningResults-Set" + str(testNo) + "/ALPool/ShapeFeaturesIV"
  fN = "ResultsRandom/ActiveLearningResults-Set" + str(testNo) + "/" + str(kWord) + "/ObjectFeaturesIII"
# fN = "ALLL" + str(testNo) + "/ML/Shape"
#fN = "ActiveLearningResults-Set3/ALPool/RGBFeaturesI"
#fN = "ActiveLearningResults-Set3/ML/ObjectFeaturesI"
#fN = "ActiveLearningResults-Set0/ML/RGBFeaturesI"
#ALUncertainity
  d = os.listdir(fN + "/NoOfDataPoints/")
  testID = " " + str(testNo) + " "
  script = "cLL-ML.py"
  if kWord == 'ALPool':
    script = "cLL-AL-Pool.py"
  elif kWord == 'ALUnc':
    script = "cLL-AL-UnC.py"
  category = "object"
  missNo = []
#for i in [4890, 5030, 5040, 5050, 5060, 5070, 5100]:
  for i in np.arange(10,6060,10):
# for i in [6050]:
   fDirName = fN + "/NoOfDataPoints/" + str(i) + "/groundTruthPrediction.csv"
#  if  (str(i) not in d) or (os.stat(fDirName).st_size==0):
#   if  (str(i) not in d):
#   if not os.path.exists(fDirName):
   if (os.stat(fDirName).st_size==0):   
    cmd = "#!/bin/sh\n#SBATCH --job-name="+ str(kWord) + str(testNo) + "-" + str(i)+"\n#SBATCH --output="+  str(kWord) + str(testNo) + "-" +str(i) +".out\n"
    cmd += "#SBATCH --error="+ str(kWord) + str(testNo) + "-" +str(i) + ".err\n#SBATCH --mem=20000\n#SBATCH --partition=batch\n"
    cmd += "#SBATCH --qos=short\n";

    cmd += "\n\npython " + script + " " + fN +" " + str(i) + testID + " " + category
    print cmd
    fName = "code-"+ str(kWord) + str(testNo) + "-" +str(i) + ".slurm"
    f = open(fName,"w")
    f.write(cmd)
    f.close()
    os.system("sbatch " + fName)
    missNo.append(i)
#  print missNo
os.system("squeue |grep npillai1")  
