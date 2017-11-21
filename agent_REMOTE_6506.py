import os
import numpy
import hdf5_getters
import pyBN
from sklearn import metrics, cluster
from sklearn.cluster import MeanShift, estimate_bandwidth, AffinityPropagation, KMeans, DBSCAN
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, StandardScaler


def featureExtract(ds_path):
	lst=open(ds_path + "list.txt",'r') 
	rows = lst.readlines()
	lst.close;
	features=numpy.matrix([1]*7)
	for line in rows:
		hdf5path = os.path.join(ds_path, line[:-1])
		songidx = 0
		onegetter = ''
		# sanity check
		if not os.path.exists(hdf5path):
			print ('ERROR: file ' + hdf5path +'does not exist.')
			sys.exit(0)
		h5 = hdf5_getters.open_h5_file_read(hdf5path)
		numSongs = hdf5_getters.get_num_songs(h5)
		if songidx >= numSongs:
			print('ERROR: file contains only ' + numSongs)
			h5.close()
			sys.exit(0)
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

	print n_clusters_
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
		prob.append(1/(numpy.linalg.norm(featureVector-x)));
	total_prob=numpy.sum(prob);
	for idx,p in enumerate(prob):
		prob[idx]=p/total_prob;
	return prob
	
def createModel(probabilities):
	model=pyBN.Network('SongRecommender');
	Fu=node('Fu');
	Fu.addOutcome(['yes','no'])
	
	return model


def updateModel():
	return 0
	
def predict(model):
	return 0
