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


def songNames(songs):
	if(numpy.size(songs)>1):
		names=[];
		for songpath in songs:
			h5 = hdf5_getters.open_h5_file_read(songpath)
			names.append(hdf5_getters.get_title(h5));
			h5.close()
	elif(numpy.size(songs)==1):
		h5 = hdf5_getters.open_h5_file_read(songs)
		names=hdf5_getters.get_title(h5);
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
	
def createModel(probabilities,names, centroids):
	model=Network('SongRecommender');
	clusterPredict=Node('clusterPredict');
	k=1
	for i in centroids:
		clusterPredict.addOutcome('c'+str(k));                                                 
		k=k+1                                                                                  
	clusterPredict.setProbabilities([1./len(centroids)]*len(centroids)) 
	model.addNode(clusterPredict);
	song_nodes=[];
	arc_nodes=[];
	i=0;
	for p in probabilities:
		n=Node('song_'+str(i+1)+'_'+names[i]);
		n.addOutcomes(['recommended','notRecommended'])
		tmp=[];
		for value in p:
			tmp.append(value)
			tmp.append(1-value)
			
		n.setProbabilities(tmp)
		a=Arc(clusterPredict,n);
		model.addNode(n);		
		song_nodes.append(n);
		arc_nodes.append(a);
		i=i+1;
	return model


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
		n_max.append(numpy.max(p_truth));
	ind=numpy.argsort(n_max)
	return ind[:10]
