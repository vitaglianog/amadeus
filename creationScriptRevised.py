import os
import sys
import numpy
import agentRevised
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
features=agentRevised.featureExtract(songs,0);	
print "finished scaling"

#prefiltering goes here

centroids = agentRevised.clustering(features,5);
n_clusters=len(centroids);
numpy.savetxt('centroids.txt', centroids)
print "finished clustering: saving centroids"
#print centroids
c = open('centroids.pckl', 'wb')
pickle.dump(centroids, c)
c.close()

names=agentRevised.songNames(songs);

f = open('song_features.pckl', 'wb')
pickle.dump(features, f)
f.close()

#model= agent.createModel(prob,names, centroids);
#save python object
#f = open('model.pckl', 'wb')
#pickle.dump(model, f)
#f.close()

#save genie file
#model.writeFile('model.xdsl');

print "Bayesian Network created"
print 'finished correctly'
