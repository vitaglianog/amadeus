import os
import sys
import numpy
import agentRevised
import pickle
import agent_pre
from sklearn.preprocessing import scale
from random import randint

numpy.set_printoptions(suppress=True)

ds_path='lib/cal500/data/';
#numpy.savetxt('features.txt', features)
lst=open(ds_path + "list.txt",'r') 
rows = lst.readlines()
lst.close;
songs=[];
for line in rows:
	songs.append(os.path.join(ds_path, line[:-1]))

#Actual listened songs
listenedSongs=[];
for i in range(10):
	listenedSongs.append(songs[randint(0,502)]);
names=agentRevised.songNames(listenedSongs);
print "\nWelcome to Amadeus:\n You have listened these songs: \n"
for n in names:
	print n

print "\nInitializing bayesian module...\n"
c_file=open('centroids.pckl','rb');
centroids = pickle.load(c_file);
c_file.close();
model=agentRevised.createModel(listenedSongs,centroids);

print "\nAmadeus is computing your recommendations...\n"
f_file= open('song_features.pckl', 'rb')
features = pickle.load(f_file);
f_file.close();
utilities=[];

[features,del_ind]=agentRevised.prefiltering(features);
for idx in del_ind:
	songs=numpy.delete(songs,idx,0);
print "Number of songs after prefiltering: " + str(len(features))

for f in features:
	p=agentRevised.dist2prob(f,centroids);
	utilities.append(agentRevised.computeUtility(model,p));
ind_recomm=agentRevised.selectBestSongs(utilities);

print "\nRecommended songs, from the most to the least:\n"
for i in reversed(ind_recomm):
	print str(agentRevised.songNames(songs[i]));
