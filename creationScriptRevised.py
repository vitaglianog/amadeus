import os
import sys
import numpy
import agentRevised
import pickle

from lib import *

from sklearn.preprocessing import scale

#ds_path='lib/cal500/data/';
ds_path='lib/MillionSongSubset/data/';

print "Acquiring dataset..."
lst=open(ds_path + "list.txt",'r') 
rows = lst.readlines()
lst.close;
songs=[];
for line in rows:
	songs.append(os.path.join(ds_path, line[:-1]))

print "Performing features extraction..."
features=agentRevised.featureExtract(songs,0);	
print "finished scaling"
f = open(ds_path+'song_features.pckl', 'wb')
pickle.dump(features,f)
f.close()

print "Performing clustering..."
centroids = agentRevised.clustering(features,12);
n_clusters=len(centroids);
c = open(ds_path+'centroids.pckl', 'wb')
pickle.dump(centroids, c)
c.close()


#model= agent.createModel(prob,names, centroids);
#save python object
#f = open('model.pckl', 'wb')
#pickle.dump(model, f)
#f.close()

#save genie file
#model.writeFile('model.xdsl');

print "Bayesian Network created"
print 'finished correctly'
