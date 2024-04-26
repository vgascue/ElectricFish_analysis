# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:16:02 2023

@author: Neuropixel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.spatial import distance
from scipy.stats import zscore
from scipy.signal import find_peaks

sf = 50000;
n_channels = 3;
folders = ['E:/EI simmulations/raw/Fish ' + str(i) + '/' for i in range(1,7)]
n=4
for folder in folders:
    os.chdir(folders[5])
    VIDEOfiles = sorted(glob.glob('*h5'))
    EODfiles = sorted(glob.glob('*bin'))
    obj_file = glob.glob('*txt')
    obj_coordinates = np.loadtxt(obj_file[0], delimiter=' ')
    #define time vectors
    if n<4:
        fs = 149.57416666666666;
    else:
        fs = 50.06166666666667; 
    n += 1
    
    selected_frames = pd.DataFrame(np.zeros((0,1)))
    for i in range(1,len(VIDEOfiles)):
        track = pd.read_hdf(VIDEOfiles[i])
        EOD = np.fromfile(EODfiles[5],dtype=np.float64)
        EOD_ch = EOD.reshape((round(len(EOD)/n_channels),n_channels))
        del EOD
        
        durationEOD = 20 * 60 #in seconds
        durationVideo = len(track)/fs 
        videoTime = np.linspace(0, round(durationVideo), round(durationVideo*fs))
        EODTime = np.linspace(0, durationEOD, durationEOD*sf)
        
        bodyparts = np.array(['chin', 'mouth', 'head', 'body1', 'body2', 'tail'])
        xpositions = pd.DataFrame(np.zeros((len(track),6)), columns=bodyparts)
        for i in range(6):
            xpositions[bodyparts[i]] = (track[track.columns.get_level_values(0)[0], bodyparts[i], 'x'])
        median_xposition = np.median(xpositions, axis=1)
        ypositions = pd.DataFrame(np.zeros((len(track),6)), columns=bodyparts)
        for i in range(6):
            ypositions[bodyparts[i]] = (track[track.columns.get_level_values(0)[0], bodyparts[i], 'y'])
        median_yposition = np.median(ypositions, axis=1)
        
        fishdistance = np.zeros(shape=(len(median_xposition),3))
        for i in range(len(median_xposition)):
            fishdistance[i,0]=median_xposition[i]/12
            fishdistance[i,1]=median_yposition[i]/12
            fishdistance[i,2]=distance.euclidean(obj_coordinates/12, [fishdistance[i,0],fishdistance[i,1]])
        
        time_close = videoTime[fishdistance[:,2] < 15]
     
        #Pre-processing to level the noise
        medianCh0 = np.median(EOD_ch[:,0])
        medianCh1 = np.median(EOD_ch[:,1])
        
        EOD_ch[:,0] -= medianCh0
        EOD_ch[:,1] -= medianCh1
        
        # extract the times of stimulation 
        Object_on = EOD_ch[:,2] > 1 
        
        # merge first to second channels to get fish's signal and plot
        EOD = EOD_ch[:,0]**4 + EOD_ch[:,1]**4;

        # calculate the z-score 
        z_score = zscore(EOD)
        
        # threshold for finding EOD peaks
        threshold = 1
        EOD_peaks, _ = find_peaks(z_score, height = threshold, distance=300)

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
       
        time_objOn = np.array([])
        on = findOn(Object_on[a:])   

        #Select 10 seconds before on and after Off
        first_on_range = [on, on+4*sf]
        stimuli1_peaks = PeaksOnRange(EOD_peaks, first_on_range[0], first_on_range[1])    
        if any([x for x in time_close if x > first_on_range[0]/sf and x < first_on_range[1]/sf]):
            stimuli1_time = EODTime[stimuli1_peaks] - on/sf
            time_objOn = np.append(time_objOn, stimuli1_time)
    
        
        for j in range(m-1):
            on += findOn(Object_on[on+200*sf:]) + 200*sf
            rangeOn = [on, on+4*sf]
            peaks = PeaksOnRange(EOD_peaks, rangeOn[0], rangeOn[1])
            if any([x for x in time_close if x > rangeOn[0]/sf and x < rangeOn[1]/sf]):
                timeOn = EODTime[peaks] - on/sf
                time_objOn = np.append(time_objOn, timeOn)
        
        selected_frames = selected_frames.append(track.iloc[time_objOn, :])
        
data = selected_frames.iloc[:,1:]
data.to_csv('SelectedFrames_fish1.csv')       
