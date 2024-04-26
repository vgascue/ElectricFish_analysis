# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:54:36 2023

@author: Neuropixel
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
fs = 149.57416666666666
#We get the coordinates for the object
obj_coordinates = np.loadtxt('ObjectCoordinates.txt', delimiter=' ')
#read our table
freqByFrame = pd.read_csv('freqByFrame.csv', header=0)

def plot_map(grid, objCoordinates,cmap, vmax, label, filename):
    fig, ax = plt.subplots()
    plt.imshow(grid, cmap=cmap, vmax=vmax, origin='lower')
    cbar = plt.colorbar()
    cbar.set_label(label)
    plt.scatter(objCoordinates[0]/12, objCoordinates[1]/12, s=100, c='k')
    fig.savefig(filename, format='svg', dpi=1200)


#Plots by hour
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
        
    timeTotal = len(freqByFrame.iloc[:round(3600*fs),:])/fs
    time_grid[time_grid == 0] = ['nan']
    freq_grid[freq_grid == 0] = ['nan']
       
    time_grid = time_grid/fs
    time_grid = time_grid/timeTotal*100
    grid3 = freq_grid*time_grid
    
    name = 'samplingDensityMap'+ str(k) + '.svg'
    #plot_map(time_grid, obj_coordinates, 'coolwarm', np.nanmax(time_grid)*0.8, 'Time Spent (%)', 'visitDensityMap17.svg')
    #plot_map(freq_grid, obj_coordinates, 'coolwarm', np.nanmax(freq_grid)*0.8, 'Median Frequency (Hz)', 'freqMap17.svg')
    plot_map(grid3, obj_coordinates, 'coolwarm', None, 'Sampling Density', name)   

#First 4 hours
grouped_freqWithObject = freqByFrame.iloc[:2153867].groupby(['x','y'])
freq_median = grouped_freqWithObject.median().reset_index()
timeDensity_count =  grouped_freqWithObject.count().reset_index()
freq_grid = np.zeros(shape=(63,61))
time_grid =  np.zeros(shape=(63,61))

for i in range(len(freq_median)):
    coords = [int(freq_median.iloc[i,0])-1,int(freq_median.iloc[i,1])-1]
    time_grid[coords[0], coords[1]] = timeDensity_count.iloc[i,5]
    freq_grid[coords[0], coords[1]] = freq_median.iloc[i,4]
    
    
time_grid[time_grid == 0] = ['nan']
freq_grid[freq_grid == 0] = ['nan']

timeTotal = len(freqByFrame)/fs 
time_grid = time_grid/fs
time_grid = time_grid/timeTotal*100
grid3 = freq_grid*time_grid

plot_map(time_grid, obj_coordinates, 'coolwarm', None, 'Time Spent (%)', 'visitDensityMap.svg')
plot_map(freq_grid, obj_coordinates, 'coolwarm',None, 'Median Frequency (Hz)', 'freqMap.svg')
plot_map(grid3, obj_coordinates, 'coolwarm', 2, 'Sampling Density', 'samplingDensityMap.svg')   
