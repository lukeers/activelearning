import os

#classes = ['red','brown','yellow','green','orange','purple','white','blue']
classes = ['red','yellow','green','blue']
#indices = [7,0,5,1,2,3,6,4]
indices = [0,1,2,3]
#indices = [7,0]
#for cls in classes:
#for cls in ['purple']:
for i in indices:
   cls = classes[i]
   cls2 = []
   for ind in indices:
       cls1 = classes[ind]
       if cls1 != cls:
           cls2.append(cls1)
#           cls2 = [cls1]
           st = " ".join(cls2)
           os.system("python testRealDataSet.py " + cls + " " + st)
#os.system("python testRealDataSet.py purple red blue")   
