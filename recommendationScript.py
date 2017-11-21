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
for line in rows[:10]:
	songs.append(os.path.join(ds_path, line[:-1]))

features=agent.featureExtract(songs,0);
feat_row=numpy.transpose(features);	
prob=[];
mean_features=[];

for f in feat_row:
	print f
	print "mean"
	print numpy.mean(f)
	print '\n'
	mean_features.append(numpy.mean(f));	

centroids=numpy.loadtxt('centroids.txt')
mean_features=scale(mean_features);
prob=agent.dist2prob(mean_features,centroids);
m_file= open('model.pckl', 'rb')
model = pickle.load(m_file)
m_file.close()
model=agent.updateModel(model,prob);
#save python object
f = open('/home/gerardo/Scrivania/AI_Project/amadeus/model.pckl', 'wb')
pickle.dump(model, f)
f.close()
#save genie file
model.writeFile('model.xdsl');

#agent.predict([]);
print prob	


