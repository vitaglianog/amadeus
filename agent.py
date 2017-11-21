import os
import numpy
import hdf5_getters
from operations import *
from network import *

from sklearn import metrics
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler


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

def clustering(data,n_clusters):
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
	model=Network('SongRecommender');
	clusterPredict=Node('clusterPredict');
	clusterPredict.addOutcomes(['c1','c2','c3','c4','c5','c6','c7','c8']);
	clusterPredict.setProbabilities([0.125]*8)
	model.addNode(clusterPredict);
	song_nodes=[];
	arc_nodes=[];
	i=0;
	for p in probabilities:
		n=Node('song'+str(i))
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
		print 'finish iteration ' + str(i)
		print 'tmp(' +str(i)+')'
		print tmp
	
	model.writeFile('model.xdsl');
	return model


def updateModel():
	return 0
	
def predict(model):
	return 0
