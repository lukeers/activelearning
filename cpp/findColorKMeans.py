# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
import numpy as np
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())
 
# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image1 = cv2.imread(args["image"])
#image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# reshape the image to be a list of pixels
image1 = image1.reshape((image1.shape[0] * image1.shape[1], 3))
# cluster the pixel intensities
clt = KMeans(n_clusters = args["clusters"])
clt.fit(image1)
# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = utils.centroid_histogram(clt)
#bar = utils.plot_colors(hist, clt.cluster_centers_)
 
highestPer = 0.0
colors = (0,0,0)
for (percent,color) in zip(hist,clt.cluster_centers_):
   # plot the relative percentage of each cluster
   print percent,color
   (r,g,b) = color
   if(((r > 20.0) and (g > 20.0)  and (b > 20.0)) or ((abs(r-g) > 15.0) or (abs(g-b) > 15.0) or (abs(r-b) >15.0))) :
      if (percent > highestPer):
         highestPer = percent
         colors = (r,g,b)

print colors

# show our color bart
fig = plt.figure()
plt.imshow(hist)
plt.show()
print "\n"

