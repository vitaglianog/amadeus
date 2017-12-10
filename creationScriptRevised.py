import os
import sys
import numpy
import agentRevised
import pickle

from lib import *

from sklearn.preprocessing import scale

#ds_path='lib/cal500/data/';
ds_path='lib/MillionSongSubset/data/';
#numpy.savetxt('features.txt', features)
lst=open(ds_path + "list.txt",'r') 
rows = lst.readlines()
lst.close;
songs=[];
for line in rows:
	songs.append(os.path.join(ds_path, line[:-1]))

print "finished acquiring dataset"

#uncomment for first execution 
features=agentRevised.featureExtract(songs,0);	
print "finished scaling"
#print features
f = open(ds_path+'song_features.pckl', 'wb')
pickle.dump(features,f)
f.close()

f = open(ds_path+'song_features.pckl','rb')
features=pickle.load(f)
f.close()
print "finished extracting features"

centroids = agentRevised.clustering(features,12);
n_clusters=len(centroids);
numpy.savetxt(ds_path+'centroids.txt', centroids)
print "finished clustering: saving centroids"
#print centroids
c = open(ds_path+'centroids.pckl', 'wb')
pickle.dump(centroids, c)
c.close()

names=agentRevised.songNames(songs);



#model= agent.createModel(prob,names, centroids);
#save python object
#f = open('model.pckl', 'wb')
#pickle.dump(model, f)
#f.close()

#save genie file
#model.writeFile('model.xdsl');

print "Bayesian Network created"
print 'finished correctly'
