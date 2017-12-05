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

number_songs= len(features)
print "number of songs: ", number_songs

print "Choose Context"

print "Time of the day:"
print "1. Morning"
print "2. Afternoon"
print "3. Evening"
print "4. Night"

time_day = raw_input()

#
# fazer um while para os erros
#
if( (time_day =="1") or (time_day =="2") or (time_day =="3" ) or (time_day =="4")):
	time_day = float(time_day)
else:
	print "you have to write one of the number options"
	
	print "Time of the day:"        # tempo row_features[5]
	print "1. Morning"
	print "2. Afternoon"
	print "3. Evening"
	print "4. Night"
	time_day = raw_input()

print "Kind of week:"               # danceability row_features[0]
print "1. Working"
print "2. Weekend"
print "3. Holiday"

week = raw_input()

#
# fazer um while para os erros
#
if( week == "1" or week =="2" or week =="3"):
		week = float(week)
else:
	print "you have to write one of the number options"
	print "Kind of week:"
	print "1. Working"
	print "2. Weekend"
	print "3. Holiday"
	week = raw_input()

print "Season:"                     # loudness  row_features[2]
print '1. Winter'
print '2. Spring'
print '3. Summer'
print '4. Autumn'

season = raw_input()

#
# fazer um while para os erros
#
if( season == "1" or season =="2" or season =="3" or season =="4"):
		season = float(season)
else:
	print "you have to write one of the number options"
	print "Season:"
	print '1. Winter'
	print '2. Spring'
	print '3. Summer'
	print '4. Autumn'
	season = raw_input()
	
songs_deleted = 0
s=0
to_delete = []
for song in features:
	if time_day == 1:  #morning
		if week == 1: #work
			if season == 1: #winter
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.25:
							to_delete.append(s)
					elif i == 0:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 2:
						if song[i] > 0.8:
							to_delete.append(s)
			elif season == 2: #spring
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.25:
							to_delete.append(s)
					elif i == 0:
						if song[i] > 0.75:
							to_delete.append(s)	
					elif i==2:
						if song[i] < 0.1:
							to_delete.append(s)
			elif season == 3: #summer
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.25:
							to_delete.append(s)
					elif i == 0:
						if song[i] > 0.75:
							to_delete.append(s) 	
					elif i == 2:
						if song[i] < 0.2:
							to_delete.append(s)
			elif season == 4: #autumn
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.25:
							to_delete.append(s)
					elif i == 0:
						if song[i] > 0.75:
							to_delete.append(s)	
					elif i == 2:
						if song[i] > 0.9:
							to_delete.append(s)
			else:
				print "ERROR on season"
				
		elif week == 2: #weekend
			if season == 1: #winter
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.25:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 2:
						if song[i] > 0.8:
							to_delete.append(s)
			elif season == 2: #spring
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.25:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 2:
						if song[i] < 0.1:
							to_delete.append(s)
			elif season == 3: #summer
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.25:
							v
					elif i == 0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 2:
						if song[i] < 0.2:
							to_delete.append(s)
			elif season == 4:	#autumn
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.25:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 2:
						if song[i] > 0.9:
							to_delete.append(s) 
			else:
				print "ERROR on season"	
				
		elif week == 3: #holiday	
			if season == 1: #winter
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.25:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.3:
							to_delete.append(s)
					elif i == 2:
						if song[i] > 0.8:
							to_delete.append(s) 
			elif season == 2: #spring
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.25:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.3:
							to_delete.append(s)
					elif i == 2:
						if song[i] < 0.1:
							to_delete.append(s) 
			elif season == 3: #summer
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.25:
							to_delete.append(s) 
					elif i == 0:
						if song[i] < 0.3:
							to_delete.append(s) 
							break
					elif i == 2:
						if song[i] < 0.2:
							to_delete.append(s)
							
			elif season == 4:	#autumn
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.25:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.3:
							to_delete.append(s)
					elif i == 2:
						if song[i] > 0.9:
							to_delete.append(s)
			else:
				print "ERROR on season"	
		else:
			print "ERROR on week"
				
				
	elif time_day == 2: #afternoon
		if week == 1: #work
			if season == 1: #winter
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 0:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 2:
						if song[i] > 0.8:
							to_delete.append(s) 
			elif season == 2: #spring
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 0:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 2:
						if song[i] < 0.1:
							to_delete.append(s)
			elif season == 3: #summer
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 0:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 2:
						if song[i] < 0.2:
							to_delete.append(s)
			elif season == 4:	#autumn
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 0:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 2:
						if song[i] > 0.9:
							to_delete.append(s)
			else:
				print "ERROR on season"
							
		elif week == 2: #weekend
			if season == 1: #winter
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 2:
						if song[i] > 0.8:
							to_delete.append(s)
			elif season == 2: #spring
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 2:
						if song[i] < 0.1:
							to_delete.append(s)
			elif season == 3: #summer
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i ==2:
						if song[i] < 0.2:
							to_delete.append(s)
			elif season == 4:	#autumn
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i==2:
						if song[i] > 0.9:
							to_delete.append(s)
			else:
				print "ERROR on season"
				
		elif week == 3: #holiday
			if season == 1: #winter
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.15:
							to_delete.append(s)	
					elif i == 0:
						if song[i] < 0.3:
							to_delete.append(s)
					elif i ==2:
						if song[i] > 0.8:
							to_delete.append(s)
			elif season == 2: #spring
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.3:
							to_delete.append(s)
					elif i == 2:
						if song[i] < 0.1:
							to_delete.append(s)
			elif season == 3: #summer
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.3:
							to_delete.append(s)
					elif i==2:
						if song[i] < 0.2:
							to_delete.append(s)
			elif season == 4:	#autumn
				for i in range(len(song)):
					if i == 5:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.3:
							to_delete.append(s)
					elif i ==2:
						if song[i] > 0.9:
							to_delete.append(s)
			else:
				print "ERROR on season"	
		else:
			print "ERROR on week"
			
			
	elif time_day == 3: #evening
		if week == 1: #work
			if season == 1: #winter
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.85:
							to_delete.append(s)
					elif i ==0:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i ==2:
						if song[i] > 0.8:
							to_delete.append(s)
			elif season == 2: #spring
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.85:
							to_delete.append(s)
					elif i == 0:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i==2:
						if song[i] < 0.1:
							to_delete.append(s)
			elif season == 3: #summer
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.85:
							to_delete.append(s)
					elif i == 0:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i ==2:
						if song[i] < 0.2:
							to_delete.append(s)
			elif season == 4:	#autumn
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.85:
							to_delete.append(s)
					elif i ==0:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i==2:
						if song[i] > 0.9:
							to_delete.append(s)
			else:
				print "ERROR on season"
				
		elif week == 2: #weekend
			if season == 1: #winter
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.85:
							to_delete.append(s)
					elif i==0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i==2:
						if song[i] > 0.8:
							to_delete.append(s)
			elif season == 2: #spring
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.85:
							to_delete.append(s)
					elif i ==0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i ==2:
						if song[i] < 0.1:
							to_delete.append(s) 
			elif season == 3: #summer
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.85:
							to_delete.append(s) 
					elif i==0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i==2:
						if song[i] < 0.2:
							to_delete.append(s)
			elif season == 4:	#autumn
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.85:
							to_delete.append(s)
					elif i==0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i==2:
						if song[i] > 0.9:
							to_delete.append(s)
			else:
				print "ERROR on season"
				
		elif week == 3: #holiday
			if season == 1: #winter
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.85:
							to_delete.append(s)
					elif i==0:
						if song[i] < 0.3:
							to_delete.append(s)
					elif i==2:
						if song[i] > 0.8:
							to_delete.append(s)
			elif season == 2: #spring
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.85:
							to_delete.append(s)
					elif i==0:
						if song[i] < 0.3:
							to_delete.append(s)
					elif i==2:
						if song[i] < 0.1:
							to_delete.append(s)
			elif season == 3: #summer
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.85:
							to_delete.append(s)
					elif i==0:
						if song[i] < 0.3:
							to_delete.append(s)
					elif i==2:
						if song[i] < 0.2:
							to_delete.append(s)
			elif season == 4:	#autumn
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.85:
							to_delete.append(s)
					elif i==0:
						if song[i] < 0.3:
							to_delete.append(s)
					elif i==2:
						if song[i] > 0.9:
							to_delete.append(s)
			else:
				print "ERROR on season"	
		
		else:
			print "ERROR on week"
	
			
	elif time_day == 4: #night	
		if week == 1: #work	
			if season == 1: #winter
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 0:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 2:
						if song[i] > 0.8:
							to_delete.append(s)
							
			elif season == 2: #spring
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 0:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 2:
						if song[i] < 0.1:
							to_delete.append(s)
			elif season == 3: #summer
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 0:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 2:
						if song[i] < 0.2:
							to_delete.append(s)
							
			elif season == 4:	#autumn
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 0:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 2:
						if song[i] > 0.9:
							to_delete.append(s)
			else:
				print "ERROR on season"
			
		elif week == 2: #weekend
			if season == 1: #winter
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 2:
						if song[i] > 0.8:
							to_delete.append(s)
			elif season == 2: #spring
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 2:
						if song[i] < 0.1:
							to_delete.append(s)
			elif season == 3: #summer
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 2:
						if song[i] < 0.2:
							to_delete.append(s)
			elif season == 4:	#autumn
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.15:
							to_delete.append(s)
					elif i == 2:
						if song[i] > 0.9:
							to_delete.append(s)
			else:
				print "ERROR on season"
		
		elif week == 3: #holiday
			if season == 1: #winter
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.3:
							to_delete.append(s)
					elif i == 2:
						if song[i] > 0.8:
							to_delete.append(s)
			elif season == 2: #spring
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.3:
							to_delete.append(s)
					elif i == 2:
						if song[i] < 0.1:
							to_delete.append(s)
			elif season == 3: #summer
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.3:
							to_delete.append(s)
					elif i == 2:
						if song[i] < 0.2:
							to_delete.append(s)
			elif season == 4:	#autumn
				for i in range(len(song)):
					if i == 5:
						if song[i] > 0.75:
							to_delete.append(s)
					elif i == 0:
						if song[i] < 0.3:
							to_delete.append(s)
					elif i == 2:
						if song[i] > 0.9:
							to_delete.append(s)
			else:
				print "ERROR on season"	
				
		else:
			print "ERROR on week"	
			
			
	else:
		print "ERROR on time_day"
	s=s+1
	


for song in to_delete:
	features[song] = [0,0,0,0,0,0];

i=502
while i >= 0:
	if features[i].all(0):
		features = numpy.delete(features, i, 0)
	i=i-1	
		
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
