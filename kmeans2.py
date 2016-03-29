import matplotlib.pyplot as plt
import pandas as pd
import scipy
import random
import math
import numpy as np
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq
from operator import sub
from scipy.spatial.distance import cdist,pdist
from matplotlib import cm
from scipy import cluster
from matplotlib import pyplot

data = pd.read_csv('un.csv', header=0)

data= data[data.lifeMale.notnull() & data.lifeFemale.notnull() & data.infantMortality.notnull() & data.GDPperCapita.notnull()] #removes null values

b = data[['lifeMale', 'lifeFemale', 'infantMortality', 'GDPperCapita']] #converting to numpy representation in ND frame
print "Data values"
print b
#print type(b)

for k in range (1, 11):
    centroids, distortion = kmeans(b.values, k)
    #print kmeans(b.values, k)
    
print ""
print("Centroid locations")
print centroids
#print type(centroids)

#distance between each point and each cluster centroid
centroid_distance = np.zeros((len(centroids), len(b.index)), dtype = np.object) #create matrix to store results
#print len(centroid_distance)
#print type(centroid_distance)

for i_centroid in xrange(10):
    for k, i_data in enumerate(b.index):
        datak = b.ix[i_data]
        #print datak
        #print k
        #print i_centroid
        distance = sum([(a-c)**2 for a,c in zip(datak, centroids[i_centroid])])
        #print distance
        centroid_distance[i_centroid, k] = distance

print ""
print "Centroid Distances"
print centroid_distance

print ""
print "Predicted clusters"
one, two = vq(b.values, centroids)
print one

#Within Cluster sum of squares http://stackoverflow.com/questions/6645895/calculating-the-percentage-of-variance-measure-for-k-means
KM = [kmeans(b.values,k) for k in range(1, 11)]
centroids2 = [cent for (cent,var) in KM]
D_k = [cdist(b.values, cent, 'euclidean') for cent in centroids2]
dist = [np.min(D,axis=1) for D in D_k]
cIdx = [np.argmin(D,axis=1) for D in D_k]
tot_withinss = [sum(d**2) for d in dist]  # Total within-cluster sum of squares
totss = sum(pdist(b.values)**2)/b.values.shape[0]       # The total sum of squares
betweenss = totss - tot_withinss          # The between-cluster sum of squares

print ""
print "Total within Cluster Sum of Squares"
print tot_withinss

##### plots #####
kIdx = 9        # K=10
clr = cm.spectral( np.linspace(0,1,10) ).tolist()
mrk = 'os^p<dvh8>+x.'

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1, 11), betweenss/totss*100, 'b*-')
ax.plot(range(1, 11)[kIdx], betweenss[kIdx]/totss*100, marker='o', markersize=12, 
    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
ax.set_ylim((0,100))
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained (%)')
plt.title('Elbow for KMeans clustering')

plt.figure()
plt.show()

#another example of elbow plot# http://stats.stackexchange.com/questions/9850/how-to-plot-data-output-of-clustering
initial = [cluster.vq.kmeans(b.values, k) for i in range(1,10)]
pyplot.plot([var for (cent,var) in initial])
pyplot.show()

#Plotting data with 3 clusters#
for k in range (1, 3):
    centroids3, distortion3 = kmeans(b.values, k)
    #print kmeans(b.values, k)
    assignment,cdist = cluster.vq.vq(b.values, centroids3)

plt.suptitle('Per Capita GDP x Infant Mortality', fontsize=18)
plt.xlabel('Per Capita GDP in USD', fontsize=16)
plt.ylabel('Infant Mortality, per 1000', fontsize=16)
plt.scatter( b['GDPperCapita'], b['infantMortality'], c=assignment)
plt.show()

plt.suptitle('Per Capita GDP x Male Life Expectancy', fontsize=18)
plt.xlabel('Per Capita GDP in USD', fontsize=16)
plt.ylabel('Male Life Expectancy', fontsize=16)
plt.scatter( b['GDPperCapita'], b['lifeMale'], c=assignment)
plt.show()

plt.suptitle('Per Capita GDP x Female Life Expectancy', fontsize=18)
plt.xlabel('Per Capita GDP in USD', fontsize=16)
plt.ylabel('Female Life Expectancy', fontsize=16)
plt.scatter( b['GDPperCapita'], b['lifeFemale'], c=assignment)
plt.show()

