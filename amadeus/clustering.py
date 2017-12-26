import os
import sys
import numpy
import amadeus
import pickle
import sklearn.metrics
from sklearn.preprocessing import scale
from random import randint

numpy.set_printoptions(suppress=True)
#ds_path='./dataset/cal500/';
ds_path='./dataset/mss/';

f_file= open(ds_path + 'song_features.pckl', 'rb')
features = pickle.load(f_file);
f_file.close();

print "Performing clustering..."



print "\nPerforming Meanshift clustering..."
[m_centroids,m_labels,m_nclusters]=amadeus.meanshift_clustering(features);
m_s=sklearn.metrics.silhouette_score(features, m_labels)
print "Silhouette score "+str(m_s);
print "Number of clusters "+str(m_nclusters);
c = open(ds_path+'m_centroids.pckl', 'wb')
pickle.dump(m_centroids, c)
c.close()


print "\nPerforming k-mean (3)  clustering..."
[k3_centroids,k3_labels]=amadeus.clustering(features,3);
k3_s=sklearn.metrics.silhouette_score(features, k3_labels)
print "Silhouette score "+str(k3_s);
c = open(ds_path+'k3_centroids.pckl', 'wb')
pickle.dump(k3_centroids, c)
c.close()



print "\nPerforming k-mean (4)  clustering..."
[k4_centroids,k4_labels]=amadeus.clustering(features,4);
k4_s=sklearn.metrics.silhouette_score(features, k4_labels)
print "Silhouette score "+str(k4_s);
c = open(ds_path+'k4_centroids.pckl', 'wb')
pickle.dump(k4_centroids, c)
c.close()


print "\nPerforming k-mean (5)  clustering..."
[k5_centroids,k5_labels]=amadeus.clustering(features,5);
k5_s=sklearn.metrics.silhouette_score(features, k5_labels)
print "Silhouette score "+str(k5_s);
c = open(ds_path+'5_centroids.pckl', 'wb')
pickle.dump(k5_centroids, c)
c.close()

print "\nPerforming k-mean (6)  clustering..."
[k6_centroids,k6_labels]=amadeus.clustering(features,6);
k6_s=sklearn.metrics.silhouette_score(features, k6_labels)
print "Silhouette score "+str(k6_s);
c = open(ds_path+'6_centroids.pckl', 'wb')
pickle.dump(k6_centroids, c)
c.close()

print "\nPerforming k-mean (7)  clustering..."
[k7_centroids,k7_labels]=amadeus.clustering(features,7);
k7_s=sklearn.metrics.silhouette_score(features, k7_labels)
print "Silhouette score "+str(k7_s);
c = open(ds_path+'7_centroids.pckl', 'wb')
pickle.dump(k7_centroids, c)
c.close()

print "\nPerforming k-mean (8)  clustering..."
[k8_centroids,k8_labels]=amadeus.clustering(features,8);
k8_s=sklearn.metrics.silhouette_score(features, k8_labels)
print "Silhouette score "+str(k8_s);
c = open(ds_path+'8_centroids.pckl', 'wb')
pickle.dump(k8_centroids, c)
c.close()

print "\nPerforming k-mean (9)  clustering..."
[k9_centroids,k9_labels]=amadeus.clustering(features,9);
k9_s=sklearn.metrics.silhouette_score(features, k9_labels)
print "Silhouette score "+str(k9_s);
c = open(ds_path+'9_centroids.pckl', 'wb')
pickle.dump(k9_centroids, c)
c.close()

#print "\nPerforming AF clustering..."
#[af_centroids,af_labels,af_nclusters] = amadeus.af_prop_km_clustering(features);
#a_s=sklearn.metrics.silhouette_score(features, af_labels)
#print "Silhouette score "+str(a_s);
#print "Number of clusters " + str(af_nclusters);
#c = open(ds_path+'af_centroids.pckl', 'wb')
#pickle.dump(af_centroids, c)
#c.close()


print "Clustering finished correctly"
