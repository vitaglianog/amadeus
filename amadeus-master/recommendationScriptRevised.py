import os
import sys
import numpy
import agentRevised
import pickle
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
p_file= open('song_probabilities.pckl', 'rb')
prob = pickle.load(p_file);
p_file.close();
utilities=[];

for p in prob:
	utilities.append(agentRevised.computeUtility(model,p));
print len(utilities)
ind_recomm=agentRevised.selectBestSongs(utilities);

print "\nRecommended songs:\n"
for i in ind_recomm:
	print agentRevised.songNames(songs[i])
