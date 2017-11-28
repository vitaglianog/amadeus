import os
import sys
import numpy
import agent

from lib import *

from sklearn.preprocessing import scale

ds_path='cal500/data/';
#numpy.savetxt('features.txt', features)
#print "finished acquiring dataset"
features=agent.featureExtract(ds_path);	
data=scale(features)
n_clusters=8;
centroids = agent.clustering(data,n_clusters);
centroids1 = agent.meanshift_clustering(data);
centroids2 = agent.af_prop_km_clustering(data);
#print labels
print centroids

for song in data[:10]:
	prob=agent.dist2prob(song,centroids);
	print prob;

print 'finished correctly'
