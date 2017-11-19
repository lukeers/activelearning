import numpy as np
import cv2
import sys

arg = sys.argv
del arg[0]

fileName = arg[0]
print fileName
 
img = cv2.imread(fileName,0)
ret,thresh=cv2.threshold(img,127,255,0)
im2,contours,hie = cv2.findContours(thresh,1,2)

actCnt = contours[0];
lrgCnt = 0.0
for index in range(len(contours)) :
   cnt = contours[index]
   area = cv2.contourArea(cnt)
   print area
   if(area > lrgCnt) :
      lrgCnt = area
      actCnt = cnt

x,y,w,h = cv2.boundingRect(actCnt)
img2 = cv2.imread(fileName)  
img3 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('image',img3)  
cv2.waitKey(0)
cv2.destroyAllWindows()
print "\n"
