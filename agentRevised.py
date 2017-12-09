import os
import numpy
import pickle

from lib import *

from sklearn import metrics
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pyBN
from sklearn import metrics, cluster
from sklearn.cluster import MeanShift, estimate_bandwidth, AffinityPropagation, KMeans, DBSCAN
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def songNames(songs):
	if(numpy.size(songs)>1):
		names=[];
		for songpath in songs:
			h5 = hdf5_getters.open_h5_file_read(songpath)
			names.append(hdf5_getters.get_title(h5)+' - '+hdf5_getters.get_artist_name(h5))
			h5.close()
	elif(numpy.size(songs)==1):
		names=[];
		h5 = hdf5_getters.open_h5_file_read(songs)
		names.append(hdf5_getters.get_title(h5)+' - '+hdf5_getters.get_artist_name(h5))
		h5.close()
	return names


def featureExtract(songs,scaling=1):
	features=numpy.matrix([1]*7)
	for songpath in songs:
		songidx = 0
		# sanity check
		#if not os.path.exists(hdf5path):
			#print ('ERROR: file ' + hdf5path +'does not exist.')
			#sys.exit(0)
		h5 = hdf5_getters.open_h5_file_read(songpath)
		#numSongs = hdf5_getters.get_num_songs(h5)
		#if songidx >= numSongs:
			#print('ERROR: file contains only ' + numSongs)
			#h5.close()
			#sys.exit(0)
		row_features=['']*8;
		row_features[0]=hdf5_getters.get_danceability(h5);
		row_features[1]=hdf5_getters.get_key(h5)*hdf5_getters.get_key_confidence(h5);
		row_features[2]=hdf5_getters.get_loudness(h5);
		row_features[3]=hdf5_getters.get_mode(h5)*hdf5_getters.get_mode_confidence(h5);
		row_features[4]=hdf5_getters.get_song_hotttnesss(h5);
		row_features[5]=hdf5_getters.get_tempo(h5);
		row_features[6]=hdf5_getters.get_time_signature(h5);
	#	row_features[7]=hdf5_getters.get_year(h5);
		features=numpy.vstack([features,row_features[0:7]]);	
		h5.close()
	features = numpy.delete(features, (0), axis=0)
	features = numpy.delete(features, (0), axis=1)
	if scaling:
		features=scale(features)
	return features

# Affinity_Propagation and K-means

def af_prop_km_clustering(data):
	af = AffinityPropagation(preference=-50).fit(data)
	cluster_centers_indices = af.cluster_centers_indices_
	labels = af.labels_

	n_clusters_ = len(cluster_centers_indices)
	
	n_samples, n_features = data.shape
	n_songs = len(data)
	sample_size = 300
	kmeans=cluster.KMeans(n_clusters_).fit(data);
	kmeans=kmeans.fit(data);
	labels = kmeans.labels_
	centroids = kmeans.cluster_centers_
	return centroids

def meanshift_clustering(data):
	# The following bandwidth can be automatically detected using
	# We can change the number of samples in each cluster
	bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=100)

	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	ms.fit(data)
	labels = ms.labels_
	centroids = ms.cluster_centers_

	labels_unique = numpy.unique(labels)
	n_clusters_ = len(labels_unique)
	print n_clusters_
	return centroids


def clustering(data, n_clusters):
	n_samples, n_features = data.shape
	n_songs = len(data)
	sample_size = 300
	kmeans=cluster.KMeans(n_clusters).fit(data);
	kmeans=kmeans.fit(data);
	labels = kmeans.labels_
	centroids = kmeans.cluster_centers_
	#print labels
	#print centroids
	return centroids
	
def dist2prob(featureVector,clusters):
	prob=[];
	for x in clusters:
		prob.append(1./(numpy.linalg.norm(featureVector-x)));
	total_prob=numpy.sum(prob);
	for idx,p in enumerate(prob):
		prob[idx]=p/total_prob;
	return prob
	
def createModel(listenedSongs, centroids):
	model=Network('SongRecommender');
	
	#compute mean probabilities of listened songs
	features=featureExtract(listenedSongs,0);
	feat_row=numpy.transpose(features);	
	listenedProb=[];
	mean_features=[];
	for f in feat_row:
		mean_features.append(numpy.mean(f));	
	mean_features=scale(mean_features);
	listenedProb=dist2prob(mean_features,centroids);
	
	
	ctxtEvidence=getContext();
	#add nodes for contextual prediction
	time=Node('time');
	time.addOutcomes(['Morning','Afternoon','Evening','Night']);
	time.setProbabilities([0.25]*4)
	model.addNode(time);
	model.setEvidence('time',ctxtEvidence[0])
	
	week=Node('week');
	week.addOutcomes(['Working','Weekend','Holiday']);
	week.setProbabilities([0.7,0.25,0.05]);
	model.addNode(week);
	model.setEvidence('week',ctxtEvidence[1])

	season=Node('season');
	season.addOutcomes(['Winter','Spring','Summer','Autumn']);
	season.setProbabilities([0.25]*4);
	model.addNode(season);
	model.setEvidence('time',ctxtEvidence[2])


	nation=Node('nation');
	nation.addOutcomes(['Italy','Spain','Portugal','France','USA']);
	nation.setProbabilities([0.20]*5);
	model.addNode(nation);
	model.setEvidence('time',ctxtEvidence[3])

	
	#add aggregator node for content-based prediction
	listenedPrediction=Node('listenedPrediction');
	k=1
	for i in centroids:
		listenedPrediction.addOutcome('c'+str(k));                                                 
		k=k+1                                                                                  
	listenedPrediction.setProbabilities(listenedProb)
	model.addNode(listenedPrediction);
	
	##add node for contextual-based prediction
	#ctxt_file=open('contextual.pckl','rb');
	#ctxtProb = pickle.load(ctxt_file);
	#ctxt_file.close();
	#contextPrediction=Node('contextPrediction');
	#k=1
	#for i in centroids:
		#contextPrediction.addOutcome('c'+str(k));                                                 
		#k=k+1                                                                                  
	#aTime=Arc(time,contextPrediction);
	#aWeek=Arc(week,contextPrediction);
	#aSeason=Arc(season,contextPrediction);
	#aNation=Arc(nation,contextPrediction);
	##stubbing
	#tmp=[0.02023608768971332,0.05733558178752108,0.06576728499156829,0.1096121416526138,0.09106239460370995,0.0387858347386172,0.1298482293423271,0.1652613827993255,0.09443507588532883,0.0387858347386172,0.02866779089376054,0.08937605396290051,0.07082630691399661]*240;
	#contextPrediction.setProbabilities(tmp);
	#model.addNode(contextPrediction);
	
	#model.computeBeliefs();
	#ctxtEvidence=contextPrediction.getBeliefs();
	
	##a=contextPrediction.getBeliefs();
	#combinedProbabilities=combine(listenedProb,ctxtEvidence,0.7);
	
	##add node for combinedprediction
	#combinedPrediction=Node('combinedPrediction');
	#k=1
	#for i in centroids:
		#combinedPrediction.addOutcome('c'+str(k));                                                 
		#k=k+1                                                                                  
	#combinedPrediction.setProbabilities(combinedProbabilities);
	#aListened=Arc(listenedPrediction,combinedPrediction);
	#aContext=Arc(contextPrediction,combinedPrediction);
	#model.addNode(combinedPrediction);
	
	#no utility node, used by function
	#save genie file
	model.writeFile('modelRevised.xdsl');
	
	return model


#stubbed method to get contextual information
def getContext():
	return [1,1,1,1];

def combine(p1,p2,alpha):
	prob=[0]*len(p1);
	for ind,p in enumerate(p1):
		prob[ind]=alpha*p1[ind]+(1-alpha)*p2[ind];
		prob[ind]=prob[ind]/2;
	total_prob=numpy.sum(prob);
	for idx,p in enumerate(prob):
		prob[idx]=p/total_prob;
	return prob*len(prob)*len(prob)
	
def computeUtility(model,p):
	#this for contextual modeling
	#beliefs=model.getNode('combinedPrediction').getBeliefs();
	beliefs=model.getNode('listenedPrediction').getBeliefs();
	u=0;
	p_sq=0;
	b_sq=0;
	#u=cosine_similarity(p,beliefs);
	for idx,b in enumerate(beliefs[:12]):	#only first "column" needed
		u+=p[idx]*beliefs[idx];
		p_sq+=numpy.square(p[idx]);
		b_sq+=numpy.square(beliefs[idx]);
	u=u/(p_sq*b_sq);
	return u
	
	
def selectBestSongs(utilities):
	ind=numpy.argsort(utilities)
	return ind[-10:]
		

def predict(model, songs):

	features=featureExtract(songs,0);
	feat_row=numpy.transpose(features);	
	prob=[];
	mean_features=[];
	for f in feat_row:
		mean_features.append(numpy.mean(f));	
		
	centroids=numpy.loadtxt('centroids.txt')
	mean_features=scale(mean_features);
	prob=dist2prob(mean_features,centroids);

	print "\nUpdating bayesian module...\n"
	model.setNodeProbability('clusterPredict11',prob)

	#Actual Prediction
	nodes=model.getNodes()
	n_max=[];
	for n in nodes:
		p=nodes[1].getProbabilities();
		p_truth=[p[0],p[2],p[4],p[6],p[8],p[10],p[12],p[14]]
		print p_truth;
		n_max.append(numpy.max(p_truth));
	ind=numpy.argsort(n_max)
	return ind[:10]
	
def askContext():
	correct=False;
	while(!correct);
		print "Choose Context"
		print "Time of the day:"
		print "1. Morning"
		print "2. Afternoon"
		print "3. Evening"
		print "4. Night"

		time_day = raw_input()
		
		if( (time_day =="1") or (time_day =="2") or (time_day =="3" ) or (time_day =="4")):
			time_day = float(time_day)
			correct=True;
		else:
			print "you have to write one of the number options"

	correct=False;
	while(!correct)
		print "Kind of week:"               # danceability row_features[0]
		print "1. Working"
		print "2. Weekend"
		print "3. Holiday"

		week = raw_input()
	
		if( week == "1" or week =="2" or week =="3"):
				week = float(week)
				correct=True;
		else:
			print "you have to write one of the number options"

	correct=False;
	while(!correct)
		print "Season:"                     # loudness  row_features[2]
		print '1. Winter'
		print '2. Spring'
		print '3. Summer'
		print '4. Autumn'
		
		season = raw_input()
		if( season == "1" or season =="2" or season =="3" or season =="4"):
				season = float(season)
				correct=True;
		else:
			print "you have to write one of the number options"
	

def prefiltering(features):
	
	#[time_day,week,season]=askContext();
	time_day=1;
	week=2;
	season=3;
		
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
		
	to_delete=numpy.unique(to_delete);
	for idx in to_delete:
		features=numpy.delete(features,idx,0);
		to_delete[:] = [x - 1 for x in to_delete]
	return [features,to_delete];
