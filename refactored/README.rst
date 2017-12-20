========================
Amadeus
========================

A content-based music recommender system with time contextual pre and post filtering.

Usage
===============

---------------------
Dataset requirement:
To run the recommendation, h5 files representing song features from the cal500 dataset (or the Million Song Dataset) should be placed in dataset/cal500/ (or dataset/mss/).
If the cal500 dataset is chosen, uncomment line 12 and comment line 11 in creation.py and recommendation.py.
 
--------------------
First run:
The configuration requires running a single time (for each dataset) the script creation.py.

-----------------------
Recommendation sample:
A sample recommendation can be obtained running the script recommendation.py, which will choose 10 random songs from the chosen dataset and print the recommendations in case no filtering, a prefiltering or a postfiltering are used. 
The time context is, by default, extracted by the actual time information of the execution.
In case further experimentation is desired, in the prefiltering and postfiltering functions of amadeus.py the line
	[time_day,week,season]=getContext();
	
can be substituted with the commented
	[time_day,week,season]=askContext();


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
