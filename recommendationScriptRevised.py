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
#ds_path='lib/MillionSongSubset/data/';

lst=open(ds_path + "list.txt",'r') 
rows = lst.readlines()
lst.close;
songs=[];
for line in rows:
	songs.append(os.path.join(ds_path, line[:-1]))

names=agentRevised.songNames(songs);
for i,n in enumerate(names):
	print songs[i]+n


f_file= open(ds_path + 'song_features.pckl', 'rb')
features = pickle.load(f_file);
f_file.close();

c_file=open(ds_path + 'centroids.pckl','rb');
centroids = pickle.load(c_file);
c_file.close();

#Actual listened songs
listenedSongs=[];
for i in range(10):
	r=randint(0,502);
	listenedSongs.append(songs[r]);
	songs=numpy.delete(songs,r,0); #avoid choosing/recommending same song
	features=numpy.delete(features,r,0);
names=agentRevised.songNames(listenedSongs);
print "\nWelcome to Amadeus:\n You have listened these songs: \n"
for n in names:
	print n

print "\nInitializing bayesian module...\n"
model=agentRevised.createModel(listenedSongs,centroids);

print "\nAmadeus is computing your recommendations...\n"
utilities=[];

for f in features:
	p=agentRevised.dist2prob(f,centroids);
	utilities.append(agentRevised.computeUtility(model,p));

ind_recomm=agentRevised.selectBestSongs(utilities);

print "\nRecommended songs, from the most to the least:\n"
for i in reversed(ind_recomm):
	print str(agentRevised.songNames(songs[i]));


print "\nPerforming pre-filtering:"
[f_features,del_ind]=agentRevised.prefiltering(features);	
f_utilities=utilities;
f_songs=songs;

for idx in reversed(del_ind):
	f_songs=numpy.delete(f_songs,idx,0);
	f_utilities=numpy.delete(f_utilities,idx,0);
print "Number of songs after prefiltering: " + str(len(f_features))
f_ind=agentRevised.selectBestSongs(f_utilities);

print "\nRecommended songs, after pre-filtering:\n"
for i in reversed(f_ind):
	print str(agentRevised.songNames(f_songs[i]));

print "\nPerforming post-filtering:"
w_utilities= agentRevised.postfiltering(features,utilities);
w_ind=agentRevised.selectBestSongs(w_utilities);

print "\nRecommendend songs after post-filtering"
for i in reversed(w_ind):
	print str(agentRevised.songNames(songs[i]));
