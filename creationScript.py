import os
import sys
import hdf5_getters
import numpy
import agent

from sklearn.preprocessing import scale

ds_path='cal500/data/';
#numpy.savetxt('features.txt', features)
lst=open(ds_path + "list.txt",'r') 
rows = lst.readlines()
lst.close;
songs=[];
for line in rows:
	songs.append(os.path.join(ds_path, line[:-1]))

print "finished acquiring dataset"
features=agent.featureExtract(songs);	
data=scale(features)
print "finished scaling"
n_clusters=8;
centroids = agent.clustering(data,n_clusters);
#print labels
print "finished clustering: centroids"
#print centroids

prob=[];
for song in data[:10]:
	prob.append(agent.dist2prob(song,centroids));

#print "probabilities computed (first 10):"
print prob[:10];

model= agent.createModel(prob);
print "Bayesian Network created"


print 'finished correctly'
