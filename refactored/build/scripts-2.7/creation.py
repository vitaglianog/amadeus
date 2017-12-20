import os
import sys
import numpy
import amadeus
import pickle

from lib import *

from sklearn.preprocessing import scale

#ds_path='../dataset/cal500/';
ds_path='../dataset/mss/';

print "Acquiring dataset..."
lst=open(ds_path + "list.txt",'r') 
rows = lst.readlines()
lst.close;
songs=[];
for line in rows:
	songs.append(os.path.join(ds_path, line[:-1]))

print "Performing features extraction..."
features=amadeus.featureExtract(songs,0);	
print "finished scaling"
f = open(ds_path+'song_features.pckl', 'wb')
pickle.dump(features,f)
f.close()

print "Performing clustering..."
centroids = amadeus.clustering(features,12);
n_clusters=len(centroids);
c = open(ds_path+'centroids.pckl', 'wb')
pickle.dump(centroids, c)
c.close()

print 'Setup finished correctly!'
