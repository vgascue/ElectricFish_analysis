# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:26:40 2023

@author: Neuropixel
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.stats import sem
import seaborn as sns
from scipy.interpolate import splev, splrep
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FixedFormatter

fs = 149.57416666666666
#We get the coordinates for the object
obj_coordinates = np.loadtxt('ObjectCoordinates.txt', delimiter=' ')
#read our table
freqByFrame = pd.read_csv('freqByFrame.csv', header=0)
mean_SD = []
std_SD = []

for k in range(16):
    grouped_freqByFrame = freqByFrame.iloc[round(3600*fs)*k:round(3600*fs)*(k+1),:].groupby(['x','y'])
    freqByFrame_median = grouped_freqByFrame.median().reset_index()
    timeDensity_count =  grouped_freqByFrame.count().reset_index()
    freq_grid = np.zeros(shape=(63,61))
    time_grid =  np.zeros(shape=(63,61))
    
    for i in range(len(freqByFrame_median)):
        coords = [int(freqByFrame_median.iloc[i,0])-1,int(freqByFrame_median.iloc[i,1])-1]
        time_grid[coords[0], coords[1]] = timeDensity_count.iloc[i,3]
        freq_grid[coords[0], coords[1]] = freqByFrame_median.iloc[i,4]
        
    timeTotal = len(freqByFrame.iloc[2333342:,:])/fs
    time_grid[time_grid == 0] = ['nan']
    freq_grid[freq_grid == 0] = ['nan']
       
    time_grid = time_grid/fs
    time_grid = time_grid/timeTotal*100
    grid3 = freq_grid*time_grid
    
    fishdistance = np.zeros(shape=(len(freqByFrame_median),3))
    for i in range(len(freqByFrame_median)):
        fishdistance[i,0]=freqByFrame_median.iloc[i,0]
        fishdistance[i,1]=freqByFrame_median.iloc[i,1]
        fishdistance[i,2]=distance.euclidean(obj_coordinates/12, [fishdistance[i,0],fishdistance[i,1]])
            
    
    for i in np.linspace(1,30,30):
        subset = fishdistance[fishdistance[:,2]<i]
        subset = subset[subset[:,2]>i-1]
        sampDens = []
        for j in range(len(subset)):
            sampDens.append(grid3[int(subset[j,0]), int(subset[j,1])])
        mean_SD.append(np.nanmean(sampDens))
        std_SD.append(sem(sampDens, nan_policy='omit'))
        
    
 
mean_SD = np.array(mean_SD)
std_SD = np.array(std_SD)
mean_SD[np.isnan(mean_SD)] = 0
std_SD[np.isnan(std_SD)] = 0
mean_SD = mean_SD.reshape((16, 30))
std_SD = std_SD.reshape((16, 30))
 
palette = sns.color_palette("cubehelix", n_colors=16).as_hex()
ticks = np.linspace(0, 16, 16)
legends = ['17 pm','18 pm', '19 pm', '20 pm', '21 pm', '22 pm', '23 pm', '00 am', '01 am', '02 am', '03 am', '04 am', '05 am', '06 am', '07 am', '08 am']
cmap = ListedColormap(palette)
norm = BoundaryNorm(ticks, len(palette))
fig, ax = plt.subplots()
for j in range(16):
    lower_bound = mean_SD[j,:] - std_SD[j,:]
    upper_bound = mean_SD[j,:] + std_SD[j,:]  
    #plt.plot(np.linspace(4,30,27),mean_SD[j,3:], c=palette[j], label=legends[j], linewidth=3)
    spl = splrep(np.linspace(4,20,16),mean_SD[j,4:20],task=0,s=15,k=4)
    x2 = np.linspace(4,20, 200)
    y2 = splev(x2,spl)
    plt.plot(x2,y2, c=palette[j], linewidth=3)
    sns.scatterplot(np.linspace(4,20,16),mean_SD[j,4:20], color=palette[j], alpha=.5)

ax.set(xlabel='Distancia al estimulo (cm)', ylabel='Sampling Density')
colorbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ticks=ticks)
colorbar.set_ticks(ticks)
colorbar.ax.yaxis.set_major_formatter(FixedFormatter(legends))
plt.show()