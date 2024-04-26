# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:25:42 2023

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
folders = ['E:/Datos/Fish' + str(i) + '/' for i in range(1,7)]
n=1
n_channels = [3, 2, 2]
sequences = pd.DataFrame(np.zeros(shape=(1, 5)), columns=['freq behavior', 'frame', 'median x', 'median y', 'distance to stimuli'])
for i in range(len(folders)):
    os.chdir(folders[0])
    cases = glob.glob('*')
    #define time vectors
    if n<4:
        fs = 149.57416666666666;
    else:
        fs = 50.06166666666667; 
    n += 1
    for j in range(len(cases)):
        os.chdir(cases[1])
        VIDEOfiles = sorted(glob.glob('*h5'))
        EODfiles = sorted(glob.glob('*bin'))
        obj_file = glob.glob('*txt')
        obj_coordinates = np.loadtxt(obj_file[0], delimiter=' ')
        del obj_file
        
        for k in range(13):
            
            track = pd.read_hdf(VIDEOfiles[k])
            EOD = np.fromfile(EODfiles[k],dtype=np.float64)
            EOD_ch = EOD.reshape((round(len(EOD)/n_channels[1]),n_channels[1]))
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
    
            
            # merge first to second channels to get fish's signal and plot
            EOD = EOD_ch[:,0]**4 + EOD_ch[:,1]**4;

            # calculate the z-score and ask for threshold
            z_score = zscore(EOD)
            plt.figure()
            plt.plot(z_score)
            threshold = 0.2
            EOD_peaks, _ = find_peaks(z_score, height = threshold, distance=300)
            
            EOD_intervals = np.diff(EODTime[EOD_peaks])
            EOD_frequencies = [1/j for j in EOD_intervals]
            
            change = []
            for i in range(1,len(EOD_frequencies)):
                diff = EOD_frequencies[i] - EOD_frequencies[i-1]
                if diff > 3 :
                    change.append(3)
                elif diff < -3:
                    change.append(1)
                else:
                    change.append(2)

            videoTable=pd.DataFrame(np.zeros(shape=(len(change), 5)), columns=['freq behavior', 'frame', 'median x', 'median y', 'distance to stimuli'])
            videoTable['freq behavior'] = change
            for l in range(len(change)):
                difference = abs(videoTime - EODTime[EOD_peaks[l]])
                closest = np.min(difference)
                index = difference.tolist().index(closest)
                videoTable.iloc[l,1] = index
                videoTable.iloc[l,2] = median_xposition[l]
                videoTable.iloc[l,3] = median_yposition[l]
                videoTable.iloc[l,4] = fishdistance[l, 2]

            sequences = sequences.append(videoTable)
            print(k)

sequences.to_csv('sequences_fish1_object.csv')
plt.plot(EODTime[EOD_peaks][:-2],change)
plt.scatter(EODTime[EOD_peaks][:-2],change,c='r')
