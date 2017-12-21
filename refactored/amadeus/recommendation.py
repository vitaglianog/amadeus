import os
import sys
import numpy
import amadeus
import pickle
from sklearn.preprocessing import scale
from random import randint

numpy.set_printoptions(suppress=True)

#ds_path='./dataset/cal500/';
ds_path='./dataset/mss/';

lst=open(ds_path + "songnames.txt",'r') 
rows = lst.readlines()
lst.close;
songs=[];
for line in rows:
	songs.append(line[:-1])

f_file= open(ds_path + 'song_features.pckl', 'rb')
features = pickle.load(f_file);
f_file.close();

c_file=open(ds_path + 'centroids.pckl','rb');
centroids = pickle.load(c_file);
c_file.close();

#Actual listened songs
rockSongs=[203,512,560,1084,1247,1417,2575,3067,3847,7364,8077,8845,8883,9121,9137];
orchestraSongs=[506,780,950,1252,1401,2029,2156,2167,3218,4057,4994,5016,5224,5767,5830,9907];


listenedFeatures=[];

print "\nWelcome to Amadeus!\n You have listened these songs: \n"
for i in reversed(orchestraSongs):
	print songs[i];
	listenedFeatures.append(features[i]);
	songs=numpy.delete(songs,i,0); #avoid choosing/recommending same song
	features=numpy.delete(features,i,0);

print "\nInitializing bayesian module...\n"
model=amadeus.createModel(listenedFeatures,centroids);

print "\nAmadeus is computing your recommendations...\n"
utilities=[];

for f in features:
	p=amadeus.dist2prob(f,centroids);
	utilities.append(amadeus.computeUtility(model,p));

ind_recomm=amadeus.selectBestSongs(utilities);

print "\nRecommended songs, from the most to the least:\n"
for i in reversed(ind_recomm):
	print songs[i];


print "\nPerforming pre-filtering:"
[f_features,del_ind]=amadeus.prefiltering(features);	
f_utilities=utilities;
f_songs=songs;

for idx in reversed(del_ind):
	f_songs=numpy.delete(f_songs,idx,0);
	f_utilities=numpy.delete(f_utilities,idx,0);
print "Number of songs after prefiltering: " + str(len(f_features))
f_ind=amadeus.selectBestSongs(f_utilities);

print "\nRecommended songs, after pre-filtering:\n"
for i in reversed(f_ind):
	print f_songs[i];

print "\nPerforming post-filtering:"
w_utilities= amadeus.postfiltering(features,utilities);
w_ind=amadeus.selectBestSongs(w_utilities);

wf_utilities= amadeus.postfiltering(f_features,f_utilities);
wf_ind=amadeus.selectBestSongs(wf_utilities);

print "\nRecommendend songs after post-filtering"
for i in reversed(w_ind):
	print songs[i];

print "\n Recommended songs after post-filtering the pre-filtered"
for i in reversed(wf_ind):
	print songs[i];

