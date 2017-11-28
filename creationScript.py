import os
import sys
import numpy
import agent
import pickle

from lib import *

from sklearn.preprocessing import scale

ds_path='lib/cal500/data/';
#numpy.savetxt('features.txt', features)
lst=open(ds_path + "list.txt",'r') 
rows = lst.readlines()
lst.close;
songs=[];
for line in rows:
	songs.append(os.path.join(ds_path, line[:-1]))

print "finished acquiring dataset"
features=agent.featureExtract(songs);	
print "finished scaling"
centroids = agent.af_prop_km_clustering(features);
n_clusters=len(centroids);
numpy.savetxt('centroids.txt', centroids)
#print labels
print "finished clustering: centroids"
#print centroids

prob=[];
for song in features:
	prob.append(agent.dist2prob(song,centroids));

names=agent.songNames(songs);
model= agent.createModel(prob,names, centroids);
#save python object
f = open('model.pckl', 'wb')
pickle.dump(model, f)
f.close()

#save genie file
model.writeFile('model.xdsl');

print "Bayesian Network created"
print 'finished correctly'
