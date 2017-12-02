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
features=agentRevised.featureExtract(songs);	
print "finished scaling"
centroids = agentRevised.af_prop_km_clustering(features);
n_clusters=len(centroids);
numpy.savetxt('centroids.txt', centroids)
print "finished clustering: saving centroids"
#print centroids
c = open('centroids.pckl', 'wb')
pickle.dump(centroids, c)
c.close()

prob=[];
for song in features:
	prob.append(agentRevised.dist2prob(song,centroids));

names=agentRevised.songNames(songs);

f = open('song_probabilities.pckl', 'wb')
pickle.dump(prob, f)
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
