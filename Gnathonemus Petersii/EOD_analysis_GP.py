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

data_folder = '/Volumes/Expansion/datos_GPetersii/datos_GPetersii/Fish5/Object/raw' #cambiar a ruta con archivos .bin
os.chdir(data_folder)
files = glob.glob( '*.bin')

print('hay ' + str(len(files)) + ' archivos')

#setear parametros
sf = 50000 #hertz
duration = 20 * 60 #en segundos

#creamos el diccionario vacio
fish = {
         'FB-DOE': {},
         'Peak-time': {}}

for i in range(len(files)):
    EOD = np.fromfile(files[i],dtype=np.float64) #En esta linea se carga uno de los archivos, para eso hay que indicar el indice del archivo que queremos cargar
    EOD_ch = EOD.reshape((int(EOD.shape[0]/3), 3))[:,:2] #cambiamos la forma de nuestros datos de vector a una matriz donde cada colmna es un canal
    
    #Pre-processing to level the noise
    medianCh0 = np.median(EOD_ch[:,0])
    medianCh1 = np.median(EOD_ch[:,1])
            
    EOD_ch[:,0] -= int(medianCh0)
    EOD_ch[:,1] -= int(medianCh1)

    # merge first to second channels to get fish's signal and plot
    EOD = np.square(EOD_ch[:,0], dtype=np.float64) + np.square(EOD_ch[:,1], dtype=np.float64)
    # calculate the z-score 
    z_score = zscore(EOD)
    del EOD_ch
    # threshold for finding EOD peaks
    threshold = 0.02
    EODTime = np.linspace(0, 20*60, len(EOD))
    EOD_peaks, _ = find_peaks(z_score, height = threshold, distance=300)
    
    EOD_intervals = np.diff(EODTime[EOD_peaks])
    EOD_frequencies = [1/j for j in EOD_intervals]
    
    #saving
    name = files[i][8:-3] 
    EOD_peaks = EOD_peaks[:len(EOD_frequencies)]
    fish['FB-DOE'][name] = [x for x in EOD_frequencies if x <60]
    fish['Peak-time'][name] = [x for i,x in enumerate(EOD_peaks, start=1) if EOD_frequencies[i-1] < 60]
    print('Termino archivo ' + str(i))

    


with open('fish5_FB-DOE.pkl', 'wb') as fp: #cambiar nombre a nombre deseado del archivo a guardar
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