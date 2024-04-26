# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:06:05 2023

@author: Neuropixel
"""

import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.signal import find_peaks
from scipy.spatial import distance

#First, we get the files names
EODfiles_names = sorted(glob.glob('*.bin'))
VIDEOfiles_names = glob.glob('*.h5')
VIDEOfiles_names = sorted(VIDEOfiles_names, key=lambda x: x.lower())
#We get the coordinates for the object
obj_coordinates = np.loadtxt('ObjectCoordinates.txt', delimiter=' ')
# define parameters
n_channels = 2;
sf = 50000; 
fs = 149.57416666666666
time_objOn = np.array([])

for l in range(0,10):
    #read the h5 file and extract chin positions to a new dataframe
    track = pd.read_hdf(VIDEOfiles_names[0])
    chin = track.iloc[:, 0:2]
    #Calculate chin to object distance for each frame
    fishdistance = []
    for i in range(len(chin)):
        fishdistance.append(distance.euclidean(obj_coordinates, [chin.iloc[i,0], chin.iloc[i,1]]))
        
    #create time vectors for video and EOD data
    durationEOD = 20 * 60 #in seconds
    durationVideo = len(track)/fs 
    videoTime = np.linspace(0, round(durationVideo), round(durationVideo*fs))
    EODTime = np.linspace(0, durationEOD, durationEOD*sf)
    
    #filter times when fish's chin is 6 cm  or less to the object 
    maxdistance = 6*12 # 1 cm is 12 pixels aprox
    time_fishClose = np.array(0)
    for i in range(len(fishdistance)):
        if fishdistance[i] < maxdistance:
            time_fishClose = np.append(time_fishClose,videoTime[i])
    
    #find EOD
    data = np.fromfile(EODfiles_names[0],dtype=np.float64)
    data_ch = data.reshape((round(len(data)/n_channels),n_channels))
    
    #Pre-processing to level the noise
    medianCh0 = np.median(data_ch[:,0])
    medianCh1 = np.median(data_ch[:,1])
    
    data_ch[:,0] -= medianCh0
    data_ch[:,1] -= medianCh1
    
    # plot raw data
    #plt.figure()
    #plt.plot(time, data_ch.T)
    #plt.xlabel('time (samples)')
    
    # extract the times of stimulation 
    Object_on = data_ch[:,2] > 1 
    
    # merge first to second channels to get fish's signal and plot
    EOD = data_ch[:,0]**4 + data_ch[:,1]**4;
    #colors = ['red', 'blue']
    #cmap = ListedColormap(colors)
    #plt.figure()
    #plt.scatter(time, EOD, c=Object_on, cmap=cmap)
    #plt.xlabel('time (s)')
    #plt.ylabel('EOD')
    
    # calculate the z-score 
    z_score = zscore(EOD)
    #plt.figure()
    #plt.plot(time, z_score)
    #plt.xlabel('time (s)')
    #plt.ylabel('z-score')
    
    # threshold for finding EOD peaks
    threshold = .1
    EOD_peaks, _ = find_peaks(z_score, height = threshold, distance=300)
    
    EOD_intervals = np.diff(EODTime[EOD_peaks])
    EOD_frequencies = [1/j for j in EOD_intervals]
    
    time_close = []
    for j in time_fishClose:
        difference = abs(EODTime - j)
        closest = np.min(difference)
        index = difference.tolist().index(closest)
        time_close.append(index)
    
    
    peaks_close = EODTime.tolist().index(time_close)
    frequencies_close = [1/np.diff(j) for j in EODTime[peaks_close]]
    
    plt.figure()
    plt.scatter(EODTime[EOD_peaks][:-1], EOD_frequencies)
    plt.plot(EODTime[EOD_peaks][:-1],EOD_frequencies)
    plt.scatter(EODTime[peaks_close][:-1], frequencies_close, c='r')
    #check if we're detecting the spikes ok
    #plt.figure()
    #plt.plot(time, data_ch[:,0], 'k', alpha=0.5)
    #plt.scatter(time[EOD_peaks], np.zeros(len(time[EOD_peaks])),color='r')
    
    
    # Raster plot
    
    def findOn(boolean):
        #This function finds the first true value on a sequence of booleans and returns the index
        for i, value in enumerate(boolean):
            if value:
                return i
        
    def PeaksOnRange(peaksVector, limit1, limit2):
         aux = peaksVector[peaksVector > limit1]
         aux2 = aux[aux < limit2]
         return aux2
    
    #Find the number of obj ons in the file
    m = 0
    if not any(Object_on[:10*sf]):
        for i in range(1,len(Object_on)):
            if not Object_on[i-1] and Object_on[i]:
                m += 1
        first_cut = False
    else:
        for i in range(4*60*sf,len(Object_on)):
            if not Object_on[i-1] and Object_on[i]:
                m += 1
        first_cut = True   
    #Find first true value
    if first_cut:
        a = 4*60*sf
    else:
        a = 0
   
    on = findOn(Object_on[a:])   
    #Select 10 seconds before on and after Off
    first_on_range = [on-1*sf, on+6*sf]
    stimuli1_peaks = PeaksOnRange(EOD_peaks, first_on_range[0], first_on_range[1])    
    if any([x for x in time_fishClose if x > first_on_range[0]/sf and x < first_on_range[1]/sf]):
        stimuli1_time = EODTime[stimuli1_peaks] - on/sf
        time_objOn = np.append(time_objOn, stimuli1_time)
    
    for j in range(m-1):
        on += findOn(Object_on[on+200*sf:]) + 200*sf
        rangeOn = [on-1*sf, on+6*sf]
        peaks = PeaksOnRange(EOD_peaks, rangeOn[0], rangeOn[1])
        if any([x for x in time_fishClose if x > rangeOn[0]/sf and x < rangeOn[1]/sf]):
            timeOn = EODTime[peaks] - on/sf
            time_objOn = np.append(time_objOn, timeOn)
        

fig, ax = plt.subplots(nrows=2, ncols=1)
k = 0
f = 0 
for i in range(1,len(time_objOn)): 
    if time_objOn[i-1] > 0 and time_objOn[i] < 0:
        ax[0].scatter(time_objOn[f:i], np.ones(len(time_objOn[f:i]))*k, s=.5, color='k')
        f = i
        k += 0.01

ax[0].scatter(time_objOn[f:], np.ones(len(time_objOn[f:]))*k, s=.5, color='k')
ax[1].hist(time_objOn, bins=40, density= True, alpha=0.5, color='k')
