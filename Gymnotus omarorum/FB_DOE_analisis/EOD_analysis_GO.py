#Este script tiene la misma funcionalidad que el exploratory_EOD pero para todos los archivos en una carpeta. Genera un diccionario con la FB-DOE y el tiempo de cada pico. 
#El diccionario que guarda es un diccionario que contiene dos diccionarios: FB-DOE y Peak-time. Cada uno de estos diccionarios cuenta con un elemento por archivo cuya key es el nombre del archivo y el valor es un vector de FB-DOE y Peak-Time de cada archivo, respectivamete.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pickle
from scipy.stats import zscore
from scipy.signal import find_peaks, butter, sosfilt

data_folder = '/Volumes/Expansion/Datos G. omarorum/Fish5/Trial 3 y 4/' #cambiar a ruta con archivos .bin
os.chdir(data_folder)
files = glob.glob( '*.bin')

print('hay ' + str(len(files)) + ' archivos')

#setear parametros
sf = 10000 #hertz
duration = 20 * 60 #en segundos

#creamos el diccionario vacio
fish = {
         'FB-DOE': {},
         'Peak-time': {}}

files = files[14:-3]
for i in range(len(files)):
    EOD = np.fromfile(files[i],dtype=np.int16)
    EOD_ch = EOD.reshape((int(EOD.shape[0]/2), 2 ))
    
    #Pre-processing to level the noise
    medianCh0 = np.median(EOD_ch[:,0])
    medianCh1 = np.median(EOD_ch[:,1])
            
    EOD_ch[:,0] -= int(medianCh0)
    EOD_ch[:,1] -= int(medianCh1)

    # merge first to second channels to get fish's signal and plot
    EOD = np.square(EOD_ch[:,0], dtype=np.int32) + np.square(EOD_ch[:,1], dtype=np.int32)
    # calculate the z-score 
    z_score = zscore(EOD)

    # threshold for finding EOD peaks
    threshold = 2
    EODTime = np.linspace(0, 20*60, len(EOD))
    EOD_peaks, _ = find_peaks(z_score, height = threshold, distance=150)
    
    EOD_intervals = np.diff(EODTime[EOD_peaks])
    EOD_frequencies = [1/j for j in EOD_intervals]
    
    #saving
    name = files[i][10:-3] 
    fish['FB-DOE'][name] = EOD_frequencies
    fish['Peak-time'][name] = EOD_peaks
    print('Termino archivo ' + str(i))

with open('fish4_2_FB-DOE.pkl', 'wb') as fp: #cambiar nombre a nombre deseado del archivo a guardar
    pickle.dump(fish, fp)


Means = []
for i in range(len(files)):
    key = list(fish['FB-DOE'].keys())[i]
    mean = np.median(fish['FB-DOE'][key])
    Means.append(mean)

#plot median FB-DOE per 20min (1 file)
plt.figure()
plt.plot(Means)
plt.scatter(range(len(Means)),Means)
plt.show()