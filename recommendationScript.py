import os
import sys
import numpy
import agent


songs=[];
lst=open("cal500/data/list.txt",'r') 
	rows = lst.readlines()
	lst.close;
	for line in rows:		
		songs.append('cal500/data/'+line[:-1]);
	features=agent.featureExtract(ds_path);	

