import os
import sys
import numpy
import agent
import pickle
from sklearn.preprocessing import scale

numpy.set_printoptions(suppress=True)

ds_path='cal500/data/';
#numpy.savetxt('features.txt', features)
lst=open(ds_path + "list.txt",'r') 
rows = lst.readlines()
lst.close;
songs=[];
for line in rows:
	songs.append(os.path.join(ds_path, line[:-1]))

names=agent.songNames(songs[30:40]);
print "\nWelcome to Amadeus:\n You have listened these songs: \n"
for n in names:
	print n

print "\nLoading bayesian module...\n"
m_file= open('model.pckl', 'rb')
model = pickle.load(m_file)
m_file.close()

print "\nAmadeus is computing your recommendations...\n"
ind_recomm=agent.predict(model,songs[30:40]);

print "\nRecommended songs:\n"
for i in ind_recomm:
	print agent.songNames(songs[i])
