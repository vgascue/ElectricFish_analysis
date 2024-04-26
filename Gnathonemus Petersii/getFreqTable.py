# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 10:10:59 2023

@author: Neuropixel
"""

import pandas as pd
import glob
import numpy as np
from scipy.stats import zscore
from scipy.signal import find_peaks
import time 

#First, we get the files names
EODfiles_names = glob.glob('*.bin')
VIDEOfiles_names = glob.glob('*.h5')
VIDEOfiles_names = sorted(VIDEOfiles_names, key=lambda x: x.lower())
EODfiles_names = sorted(EODfiles_names, key=lambda x: x.lower())
# define parameters
n_channels = 3
sf = 50000; 
fs = 149.57416666666666

freqByFrame = pd.DataFrame(np.empty(shape=(0,5)), columns=['video','time', 'x', 'y', 'freq'])

for l in range(1,len(VIDEOfiles_names)):
    start = time.time()
    track = pd.read_hdf(VIDEOfiles_names[l])
    chin = track.iloc[:, 0:2]
    
    #create time vectors for video and EOD data
    durationEOD = 20 * 60 #in seconds
    durationVideo = len(track)/fs 
    videoTime = np.linspace(0, round(durationVideo), round(durationVideo*fs))
    EODTime = np.linspace(0, durationEOD, durationEOD*sf)
    
    #find EOD
    data = np.fromfile(EODfiles_names[l],dtype=np.float64)
    data_ch = data.reshape((round(len(data)/n_channels),n_channels))
    
    #Pre-processing to level the noise
    medianCh0 = np.median(data_ch[:,0])
    medianCh1 = np.median(data_ch[:,1])
    
    data_ch[:,0] -= medianCh0
    data_ch[:,1] -= medianCh1
    
    # merge first to second channels to get fish's signal and plot
    EOD = data_ch[:,0]**4 + data_ch[:,1]**4;
    
    # calculate the z-score 
    z_score = zscore(EOD)
    
    # threshold for finding EOD peaks
    threshold = .01
    EOD_peaks, _ = find_peaks(z_score, height = threshold, distance=300)
    
    EOD_intervals = np.diff(EODTime[EOD_peaks])
    EOD_frequencies = [1/j for j in EOD_intervals]
    
    #Get de median fish position for each frame
    bodyparts = np.array([x for x in track.columns.get_level_values(1)])
    xpositions = pd.DataFrame(np.zeros((len(track),6)), columns=np.unique(bodyparts))
    for i in range(6):
         xpositions[np.unique(bodyparts)[i]] = (track[track.columns.get_level_values(0)[0], np.unique(bodyparts)[i], 'x'])
    median_xposition = np.median(xpositions, axis=1)
    ypositions = pd.DataFrame(np.zeros((len(track),6)), columns=np.unique(bodyparts))
    for i in range(6):
        ypositions[np.unique(bodyparts)[i]] = (track[track.columns.get_level_values(0)[0], np.unique(bodyparts)[i], 'y'])
    median_yposition = np.median(ypositions, axis=1)
    
    EOD_round = np.round(EOD_peaks/sf,3)
    
    freqTable = pd.DataFrame(np.zeros(shape=(len(track),5)), columns=['video','time', 'x', 'y', 'freq'])
    for x, (median_x, median_y, video_time) in enumerate(zip(median_xposition, median_yposition, videoTime)):
        freqTable.iloc[x,0] = VIDEOfiles_names[l]
        freqTable.iloc[x,2] = round(median_x/12) 
        freqTable.iloc[x,3] = round(median_y/12)
        time1 = round(video_time)
        freqTable.iloc[x,1] = time1
        difference = abs(EOD_round - video_time)
        closest = np.min(difference)
        freq_index = difference.tolist().index(closest)
        freqTable.iloc[x,4] = EOD_frequencies[freq_index-1]
    
    freqByFrame = pd.concat((freqByFrame, freqTable), axis=0)
    print(l)
    elapsed = time.time() - start
    print(elapsed)
    
#save the table
freqByFrame.to_csv('freqByFrame.csv')
