import os
import numpy
import pickle
import time
import pyBN
from lib import *
from lib import holidays

from sklearn import metrics
from sklearn import cluster
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, AffinityPropagation, KMeans, DBSCAN
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, StandardScaler


def songNames(songs):
    if (numpy.size( songs ) > 1):
        names = []
        for songpath in songs:
            h5 = hdf5_getters.open_h5_file_read( songpath )
            names.append( hdf5_getters.get_title( h5 ) + ' - ' + hdf5_getters.get_artist_name( h5 ) )
            h5.close()
    elif (numpy.size( songs ) == 1):
        names = []
        h5 = hdf5_getters.open_h5_file_read( songs )
        names.append( hdf5_getters.get_title( h5 ) + ' - ' + hdf5_getters.get_artist_name( h5 ) )
        h5.close()
    return names


def featureExtract(songs, scaling=1):
    features = numpy.matrix( [1] * 5 )
    for songpath in songs:
        songidx = 0
        # sanity check
        if not os.path.exists( songpath ):
            print ('ERROR: file ' + songpath + 'does not exist.')
            sys.exit( 0 )
        h5 = hdf5_getters.open_h5_file_read( songpath )
        row_features = [''] * 5
        row_features[0] = hdf5_getters.get_key( h5 ) * hdf5_getters.get_key_confidence( h5 )
        row_features[1] = hdf5_getters.get_loudness( h5 )
        row_features[2] = hdf5_getters.get_mode( h5 ) * hdf5_getters.get_mode_confidence( h5 )
        row_features[3] = hdf5_getters.get_tempo( h5 )
        row_features[4] = hdf5_getters.get_time_signature( h5 )
        features = numpy.vstack( [features, row_features[0:5]] )
        h5.close()
    features = numpy.delete( features, (0), axis=0 )
    if scaling:
        features = scale( features )
    return features


# Affinity_Propagation and K-means
def af_prop_km_clustering(data):
    af = AffinityPropagation( preference=-50 ).fit( data )
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len( cluster_centers_indices )
    n_samples, n_features = data.shape
    n_songs = len( data )
    sample_size = 300
    kmeans = cluster.MiniBatchKMeans( n_clusters_ ).fit( data )
    kmeans = kmeans.fit( data )
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return centroids


def meanshift_clustering(data):
    bandwidth = estimate_bandwidth( data, quantile=0.2, n_samples=100 )
    ms = MeanShift( bandwidth=bandwidth, bin_seeding=True )
    ms.fit( data )
    labels = ms.labels_
    centroids = ms.cluster_centers_
    labels_unique = numpy.unique( labels )
    n_clusters_ = len( labels_unique )
    print n_clusters_
    return centroids


def clustering(data, n_clusters):
    n_samples, n_features = data.shape
    n_songs = len( data )
    sample_size = 300
    kmeans = cluster.MiniBatchKMeans( n_clusters ).fit( data )
    kmeans = kmeans.fit( data )
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return centroids


def dist2prob(featureVector, clusters):
    prob = []
    for x in clusters:
        prob.append( 1. / (numpy.linalg.norm( featureVector - x )) )
    total_prob = numpy.sum( prob )
    for idx, p in enumerate( prob ):
        prob[idx] = p / total_prob
    return prob


def createModel(listenedFeatures, centroids):
    model = Network( 'SongRecommender' )

    # compute mean probabilities of listened songs
    feat_row = numpy.transpose( listenedFeatures )
    listenedProb = []
    mean_features = []
    for f in feat_row:
        mean_features.append( numpy.mean( f ) )
    listenedProb = dist2prob( mean_features, centroids )

    ctxtEvidence = getContext()

    # add node for content-based prediction
    listenedPrediction = Node( 'listenedPrediction' )
    k = 1
    for i in centroids:
        listenedPrediction.addOutcome( 'c' + str( k ) )
        k = k + 1
    listenedPrediction.setProbabilities( listenedProb )
    model.addNode( listenedPrediction )

    ##add nodes for contextual prediction
    # time=Node('time');
    # time.addOutcomes(['Morning','Afternoon','Evening','Night']);
    # time.setProbabilities([0.25]*4)
    # model.addNode(time);
    # model.setEvidence('time',ctxtEvidence[0])

    # week=Node('week');
    # week.addOutcomes(['Working','Weekend','Holiday']);
    # week.setProbabilities([0.7,0.25,0.05]);
    # model.addNode(week);
    # model.setEvidence('week',ctxtEvidence[1])

    # season=Node('season');
    # season.addOutcomes(['Winter','Spring','Summer','Autumn']);
    # season.setProbabilities([0.25]*4);
    # model.addNode(season);
    # model.setEvidence('time',ctxtEvidence[2])

    # ctxt_file=open('contextual.pckl','rb');
    # ctxtProb = pickle.load(ctxt_file);
    # ctxt_file.close();
    # contextPrediction=Node('contextPrediction');
    # k=1
    # for i in centroids:
    # contextPrediction.addOutcome('c'+str(k));
    # k=k+1
    # aTime=Arc(time,contextPrediction);
    # aWeek=Arc(week,contextPrediction);
    # aSeason=Arc(season,contextPrediction);
    ##stubbing
    # tmp=[0.02023608768971332,0.05733558178752108,0.06576728499156829,0.1096121416526138,0.09106239460370995,0.0387858347386172,0.1298482293423271,0.1652613827993255,0.09443507588532883,0.0387858347386172,0.02866779089376054,0.08937605396290051,0.07082630691399661]*48;
    # contextPrediction.setProbabilities(tmp);
    # model.addNode(contextPrediction);

    # model.computeBeliefs();
    # ctxtEvidence=contextPrediction.getBeliefs();

    ##a=contextPrediction.getBeliefs();
    # combinedProbabilities=combine(listenedProb,ctxtEvidence,0.7);

    ##add node for combinedprediction
    # combinedPrediction=Node('combinedPrediction');
    # k=1
    # for i in centroids:
    # combinedPrediction.addOutcome('c'+str(k));
    # k=k+1
    # combinedPrediction.setProbabilities(combinedProbabilities);
    # aListened=Arc(listenedPrediction,combinedPrediction);
    # aContext=Arc(contextPrediction,combinedPrediction);
    # model.addNode(combinedPrediction);

    # no utility node, used by function
    # save genie file

    model.writeFile( 'modelRevised.xdsl' )
    return model


# blending function for contextual modelling
def combine(p1, p2, alpha):
    prob = [0] * len( p1 )
    for ind, p in enumerate( p1 ):
        prob[ind] = alpha * p1[ind] + (1 - alpha) * p2[ind]
        prob[ind] = prob[ind] / 2
    total_prob = numpy.sum( prob )
    for idx, p in enumerate( prob ):
        prob[idx] = p / total_prob
    return prob * len( prob ) * len( prob )


def computeUtility(model, p):
    # this for contextual modeling
    # beliefs=model.getNode('combinedPrediction').getBeliefs();
    beliefs = model.getNode( 'listenedPrediction' ).getBeliefs()
    u = 0
    p_sq = 0
    b_sq = 0
    for idx, b in enumerate( beliefs[:12] ):  # only first column
        u += p[idx] * beliefs[idx]
        p_sq += numpy.square( p[idx] )
        b_sq += numpy.square( beliefs[idx] )
    u = u / (p_sq * b_sq)
    return u


def selectBestSongs(utilities):
    ind = numpy.argsort( utilities )
    return ind[-10:]


def predict(model, songs):
    features = featureExtract( songs, 0 )
    feat_row = numpy.transpose( features )
    prob = []
    mean_features = []
    for f in feat_row:
        mean_features.append( numpy.mean( f ) )
    centroids = numpy.loadtxt( 'centroids.txt' )
    mean_features = scale( mean_features )
    prob = dist2prob( mean_features, centroids )
    print "\nUpdating bayesian module...\n"
    model.setNodeProbability( 'clusterPredict11', prob )

    # Actual Prediction
    nodes = model.getNodes()
    n_max = []
    for n in nodes:
        p = nodes[1].getProbabilities()
        p_truth = [p[0], p[2], p[4], p[6], p[8], p[10], p[12], p[14]]
        print p_truth
        n_max.append( numpy.max( p_truth ) )
    ind = numpy.argsort( n_max )
    return ind[:10]


def getContext():
    day = time.strftime( "%B %d %Y" )
    pt_holidays = holidays.Portugal()
    weekday = int( time.strftime( "%w" ) )  # weekday as decimal
    if weekday in range( 1, 7 ):
        week = 1
    if weekday > 6.:
        week = 2
    if (day in pt_holidays or weekday == 0):
        week = 3

    hour = int( time.strftime( "%H" ) )  # hour as decimal

    if hour >= 0:
        hour_day = 4
    if hour >= 6:
        hour_day = 1
    if hour >= 12:
        hour_day = 2
    if hour >= 18:
        hour_day = 3

    day = time.strftime( "%j" )  # day as decimal
    # "day of year" ranges for the northern hemisphere
    spring = range( 80, 172 )
    summer = range( 172, 264 )
    fall = range( 264, 355 )
    # winter = everything else

    if day in spring:
        season = 2
    elif day in summer:
        season = 3
    elif day in fall:
        season = 4
    else:
        season = 1

    return [hour_day, week, season]


def norm(X):
    X_min = min( X )
    X_max = max( X )
    X_peak = X_max - X_min

    X[:] = [(x - X_min) / X_peak for x in X]
    return X


def prefiltering(features, time_day, week, season):

    s_features = scale( features )
    s_features = numpy.transpose( s_features )
    s_features[:] = [norm( a ) for a in s_features]
    s_features = numpy.transpose( s_features )

    to_delete = []
    for ind, song in enumerate( s_features ):
        # tempo
        if (time_day == 1 and song[3] < 0.4) or (time_day == 2 and song[3] < 0.3) or (
                        time_day == 3 and song[3] > 0.7) or (time_day == 4 and song[3] > 0.6):
            to_delete.append( ind )
            continue
        # loudness
        if (week == 1 and song[1] > 0.6) or (week == 2 and song[1] < 0.3) or (week == 3 and song[1] < 0.5):
            to_delete.append( ind )
            continue
        # mode
        if (season == 1 and song[2] > 0.6) or (season == 2 and song[2] < 0.3) or (season == 3 and song[2] < 0.4) or (
                        season == 4 and song[2] > 0.75):
            to_delete.append( ind )
            continue

    for i in reversed( to_delete ):
        features = numpy.delete( features, i, 0 )

    return [features, to_delete]


def gaussian(x, x0, sigma):
    return np.exp( -np.power( (x - x0) / sigma, 2. ) / 2. )


def postfiltering(features, utilities, time_day, week, season):

    s_features = scale( features )

    s_features = numpy.transpose( s_features )
    s_features[:] = [norm( a ) for a in s_features]
    s_features = numpy.transpose( s_features )

    # Tempo decreasingly low from morning to night
    for ind, song in enumerate( s_features ):
        if time_day == 1:
            a1 = gaussian( song[3], 0.7, 1 )
        elif time_day == 2:
            a1 = gaussian( song[3], 0.6, 1 )
        elif time_day == 3:
            a1 = gaussian( song[3], 0.4, 1 )
        else:
            a1 = gaussian( song[3], 0.3, 1 )

        # loudness low for working, higher for weekend, medium for holidays
        if week == 1:
            a2 = gaussian( song[1], 0.5, 1 )
        elif week == 2:
            a2 = gaussian( song[1], 0.8, 1 )
        else:
            a2 = gaussian( song[1], 0.65, 1 )

        # mode low for winter/autumn, high for spring/summer
        if season == 1:
            a3 = gaussian( song[2], 0.3, 1 )
        elif season == 2:
            a3 = gaussian( song[2], 0.6, 1 )
        elif season == 3:
            a3 = gaussian( song[2], 0.7, 1 )
        else:
            a2 = gaussian( song[2], 0.2, 1 )

        # alpha ranges from 0 to 1, obtained as the "equal" sum of the three contexts
        alpha = (a1 + a2 + a3) / 3
        utilities[ind] = utilities[ind] * alpha

    return utilities


def creation():
    ds_path = './dataset/mss/'

    print "Acquiring dataset..."
    lst = open( ds_path + "list.txt", 'r' )
    rows = lst.readlines()
    lst.close
    songs = []
    for line in rows:
        songs.append( os.path.join( ds_path, line[:-1] ) )

    print "Performing features extraction..."
    features = featureExtract( songs, 0 )
    print "finished scaling"
    f = open( ds_path + 'song_features.pckl', 'wb' )
    pickle.dump( features, f )
    f.close()

    print "Performing clustering..."
    centroids = af_prop_km_clustering( features )
    n_clusters = len( centroids )
    c = open( ds_path + 'centroids.pckl', 'wb' )
    pickle.dump( centroids, c )
    c.close()

    print 'Setup finished correctly!'
