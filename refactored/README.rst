========================
Amadeus
========================

A content-based music recommender system with time contextual pre and post filtering.

Usage
===============

Setup
------------------
In order to install required dependencies, use the command:
pip install -r requirements.txt

Usage (command line)
-------------------
Sample configuration: run "python recommendation.py"

Changing parameters
-------------------
The file recommendation.py is a runnable script to obtain recommendations for three different playlists.
To change the listened playlist, change the value of "playlist" variable in line 20:

playlist= jazzSongs;

One of the already defined three playlists can be used, or a new one can be defined being an array of integers ranging from 0-9999.

The contextual information will be asked to the user.
To change this behaviour and automatically get context, comment line 64 and uncomment line 65.
To use a different dataset, the cal500, uncomment line 11 and comment line 12.
To use different cluster centroids, change the name of the file in line 33 to one of the *centroids.pckl present in dataset/(chosen dataset)

Creation phase:
--------------------
To reconfigure features and clusters, configuration requires running a single time (for each dataset) the scripts featureExtraction.py and clustering.py.
To perform featureExtraction, h5 files representing song features from the cal500 dataset (or the Million Song Dataset) should be placed in dataset/cal500/ (or dataset/mss/).
If the cal500 dataset is chosen, uncomment line 11 and comment line 12 in amadeus/featureExtraction.py and amadeus/clustering.py


Dataset Copyrights
=================

------------------
Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. 
The Million Song Dataset. In Proceedings of the 12th International Society
for Music Information Retrieval Conference (ISMIR 2011), 2011.

------------------
Turnbull, D., Barrington, L., Torres, D., and Lanckriet, G. (2008). Semantic
annotation and retrieval of music and sound effects. IEEE Transactions on
Audio, Speech, and Language Processing, 16(2):467–476.
