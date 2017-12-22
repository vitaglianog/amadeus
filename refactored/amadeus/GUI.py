from tkinter import *
import sys
import numpy
import os
import pickle
from sklearn.preprocessing import scale
from random import randint
import amadeus

def creation():
    amadeus.creation()
    label = Label(root, text="DataSet Created")
    label.pack()
    canvas1 = Canvas( root, width=500, height=20 )
    canvas1.pack()
    frame2 = Frame( root )
    frame2.pack()
    button2 = Button(frame2, text="Next Step", command=step2)
    button2.pack()

def model():
    rockSongs = [203, 512, 560, 1084, 1247, 1417, 2575, 3067, 3847, 7364, 8077, 8845, 8883, 9121, 9137]
    orchestraSongs = [506, 780, 950, 1252, 1401, 2029, 2156, 2167, 3218, 4057, 4994, 5016, 5224, 5767, 5830, 9907]
    jazzSongs = [509, 797, 1108, 1180, 1837, 2134, 2138, 2174, 2361, 2336, 2378, 2430, 3131, 3482, 4504, 6140]

    variable = v.get()
    if variable == 1:
        KindMusic = rockSongs
    elif variable == 2:
        KindMusic = orchestraSongs
    else:
        KindMusic = jazzSongs

    frame4 = Frame( root )
    frame4.pack()
    label = Label( frame4, text="Loading DataSet" )
    label.pack()

    numpy.set_printoptions( suppress=True )

    # ds_path='./dataset/cal500/';
    ds_path = './dataset/mss/'

    lst = open( ds_path + "songnames.txt", 'r' )
    rows = lst.readlines()
    lst.close
    songs = []
    for line in rows:
        songs.append( line[:-1] )

    f_file = open( ds_path + 'song_features.pckl', 'rb' )
    features = pickle.load( f_file )
    f_file.close()

    c_file = open( ds_path + 'centroids.pckl', 'rb' )
    centroids = pickle.load( c_file )
    c_file.close()

    model1(KindMusic, features, centroids, songs)

def model1(KindMusic, features, centroids, songs):

    listenedFeatures = []

    for i in reversed( KindMusic ):
        listenedFeatures.append( features[i] )
        songs = numpy.delete( songs, i, 0 )  # avoid choosing/recommending same song
        features = numpy.delete( features, i, 0 )

    model = amadeus.createModel( listenedFeatures, centroids )

    canvas = Canvas( root, width=500, height=20 )
    canvas.pack()
    frame5 = Frame( root )
    frame5.pack()
    label5 = Label( frame5, text="Model Created" )
    label5.pack()

    step3(features,songs, centroids, model)

def step2():
    frame3 = Frame(root)
    frame3.pack()
    label3 = Label(frame3, text="Choose Kind of Music Listened")
    label3.pack()

    v = IntVar()
    Radiobutton( frame3, text="Rock", variable=v, value=1, command=model).pack( anchor=W )
    Radiobutton( frame3, text="Orchestra", variable=v, value=2, command= model).pack( anchor=W )
    Radiobutton( frame3, text="Jazz", variable=v, value=3,  command= model ).pack( anchor=W )

def step3(features,songs, centroids, model):

    utilities = []

    for f in features:
        p = amadeus.dist2prob( f, centroids )
        utilities.append( amadeus.computeUtility( model, p ) )

    ind_recomm = amadeus.selectBestSongs( utilities )

    frame6 = Frame( root )
    frame6.pack()
    label4 = Label( frame6, text="Recommended songs, from the most to the least:" )
    label4.pack()
    to_print = ''
    for i in reversed( ind_recomm ):
        to_print = to_print + ' \n ' + str( songs[i] )

    frame7 = Frame( root )
    frame7.pack()
    label5 = Label( frame7, text=to_print )
    label5.pack()

    canvas1 = Canvas( root, width=500, height=20 )
    canvas1.pack()
    frame8 = Frame( root )
    frame8.pack()
    label6 = Label( frame8, text="Context" )
    label6.pack( side=LEFT )
    step4(features, utilities, songs)

def step4(features, utilities, songs):
    def ask():
        y = IntVar()
        x = IntVar()
        z = IntVar()

        def end():
            hour_day = x.get()
            week = y.get()
            season = z.get()

            frame18.destroy()
            frame19.destroy()
            frame20.destroy()
            frame21.destroy()
            canvas1.destroy()

            frame17 = Frame( root )
            frame17.pack()
            label = Label( frame17, text="Context Defined" )
            label.pack()
            canvas3 = Canvas( root, width=500, height=20 )
            canvas3.pack()

            return [hour_day, week, season]

        frame18 = Frame( root )
        frame18.pack()
        theLabel = Label( frame1, text="Choose the Time of the Day:", )
        theLabel.pack()
        Radiobutton( frame18, text="Morning", variable=y, value=1 ).pack( side=LEFT )
        Radiobutton( frame18, text="Afternoon", variable=y, value=2 ).pack( side=LEFT )
        Radiobutton( frame18, text="Night", variable=y, value=3 ).pack( side=LEFT )
        Radiobutton( frame18, text="Evening", variable=y, value=4 ).pack( side=LEFT )

        frame19 = Frame( root )
        frame19.pack()
        theLabel = Label( frame19, text="Choose the Season:", )
        theLabel.pack()
        Radiobutton( frame19, text="Winter", variable=x, value=1 ).pack( side=LEFT )
        Radiobutton( frame19, text="Spring", variable=x, value=2 ).pack( side=LEFT )
        Radiobutton( frame19, text="Summer", variable=x, value=3 ).pack( side=LEFT )
        Radiobutton( frame19, text="Fall", variable=x, value=4 ).pack( side=LEFT )

        frame20 = Frame( root )
        frame20.pack()
        theLabel = Label( frame20, text="Choose the Kind of day:", )
        theLabel.pack()
        Radiobutton( frame20, text="Weekday", variable=z, value=1 ).pack( side=LEFT )
        Radiobutton( frame20, text="Weekend", variable=z, value=2 ).pack( side=LEFT )
        Radiobutton( frame20, text="Holiday", variable=z, value=3 ).pack( side=LEFT )

        frame21 = Frame( root )
        frame21.pack()
        Button1 = Button( frame21, text="Finished Context", command=end )
        Button1.pack( side=LEFT )

        canvas1 = Canvas( root, width=500, height=50 )
        canvas1.pack()

    def getContext():
        def end():
            frame22 = Frame( root )
            frame22.pack()
            label = Label( frame22, text="Context Defined" )
            label.pack()
            canvas3 = Canvas( root, width=500, height=20 )
            canvas3.pack()

            Context = amadeus.getContext()
            return Context

        frame6 = Frame( root )
        frame6.pack()
        Button1 = Button( frame6, text="Finished Context", command=end )
        Button1.pack( side=LEFT )

    def kind():
        variable = v.get()
        if variable == 1:
            [hour_day, week, season] = ask()
        else:
            [hour_day, week, season] = amadeus.getContext()

        step5(hour_day, week, season, features, utilities, songs)

    frame9 = Frame( root )
    frame9.pack()
    theLabel = Label( frame9, text="Choose the Kind of Context:" )
    theLabel.pack( side=LEFT )
    Radiobutton( frame9, text="Ask", variable=v, value=1, command=kind ).pack( anchor=W )
    Radiobutton( frame9, text="Get", variable=v, value=2, command=kind ).pack( anchor=W )

def step5(hour_day, week, season, features, utilities, songs):
    [f_features, del_ind] = amadeus.prefiltering( features, hour_day, week, season )
    f_utilities = utilities
    f_songs = songs

    for idx in reversed( del_ind ):
        f_songs = numpy.delete( f_songs, idx, 0 )
        f_utilities = numpy.delete( f_utilities, idx, 0 )

    f_ind = amadeus.selectBestSongs( f_utilities )
    frame10 =Frame(root)
    frame10.pack()
    label6 = Label(frame10, text= "\nRecommended songs, after pre-filtering:\n")
    label6.pack()

    to_print = ''
    for i in reversed( f_ind ):
        to_print = to_print + ' \n ' + str( songs[i] )

    frame11 = Frame( root )
    frame11.pack()
    label5 = Label( frame11, text= to_print )
    label5.pack()

    frame12 = Frame( root )
    frame12.pack()
    label7 = Label( frame12, text="\nPerforming post-filtering:" )
    label7.pack()

    w_utilities = amadeus.postfiltering( features, utilities, hour_day, week, season )
    w_ind = amadeus.selectBestSongs( w_utilities )

    wf_utilities = amadeus.postfiltering( f_features, f_utilities, hour_day, week, season )
    wf_ind = amadeus.selectBestSongs( wf_utilities )

    frame13 = Frame( root )
    frame13.pack()
    label8 = Label( frame13, text="\nRecommendend songs after post-filtering" )
    label8.pack()

    to_print1 = ''
    for i in reversed( w_ind ):
        to_print1 = to_print1 + ' \n ' + str( songs[i] )

    frame14 = Frame( root )
    frame14.pack()
    label5 = Label( frame14, text=to_print1 )
    label5.pack()

    frame15 = Frame( root )
    frame15.pack()
    label9 = Label( frame15, text="\n Recommended songs after post-filtering the pre-filtered" )
    label9.pack()

    to_print2 = ''
    for i in reversed( wf_ind ):
        to_print2 = to_print2 + ' \n ' + str( songs[i] )

    frame16 = Frame( root )
    frame16.pack()
    label5 = Label( frame16, text=to_print2 )
    label5.pack()


root = Tk()

scrollbar = Scrollbar( root )
scrollbar.pack( side=RIGHT, fill=Y )

mylist = Listbox( root, yscrollcommand=scrollbar.set )
for line in range( 1000 ):
    mylist.insert( END, str( line ))

mylist.pack( side=LEFT, fill=BOTH )
scrollbar.config( command=mylist.yview )


v = IntVar()

frame = Frame( root )
frame.pack()
theLabel = Label(frame, text="Welcome to Amadeus" )
theLabel.pack()
canvas = Canvas(root,width=500, height=20 )
canvas.pack()
frame1= Frame(root)
frame1.pack()
button = Button(frame1, text="Creat DataSet", command=creation)
button.pack( side=LEFT )

# mantem a janela aberta
root.mainloop()
