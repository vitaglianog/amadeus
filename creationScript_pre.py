import os
import sys
import numpy
import agent_pre
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
features=agent_pre.featureExtract(songs);	
print "finished normalizing"

# Pre filtering
features=agent_pre.prefiltering(features);

print features;

number_songs= len(features)
print "number of songs: ", number_songs


#print features;
print len(features)


print "finhished contextual prefiltering"

centroids = agent_pre.af_prop_km_clustering(features);
n_clusters=len(centroids);
numpy.savetxt('centroids.txt', centroids)
print "finished clustering: saving centroids"
#print centroids
c = open('centroids.pckl', 'wb')
pickle.dump(centroids, c)
c.close()

prob=[];
for song in features:
	prob.append(agent_pre.dist2prob(song,centroids));

names=agent_pre.songNames(songs);

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
