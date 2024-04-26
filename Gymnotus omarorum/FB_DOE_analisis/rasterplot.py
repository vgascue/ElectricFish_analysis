import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pickle
from scipy.stats import zscore
from scipy.signal import find_peaks, butter, sosfilt
from datetime import datetime, timedelta

data_folder = '/Volumes/Expansion/Datos G. omarorum/Fish2/Objeto/Trial 1  y 2/raw/' #cambiar a ruta con archivos .bin
os.chdir(data_folder)
files = glob.glob( '*.bin')
on_off = pd.read_csv("/Volumes/Expansion/Datos G. omarorum/Fish2/Objeto/Trial 1  y 2/raw/on_off_trial1.csv", header=None)
n_channels = 2
sf= 10000

files_start = []
for i in range(len(files)):
    files_start.append(datetime.strptime(files[i][10:-4], '%Y-%m-%dT%H_%M_%S'))


for i in range(on_off.shape[0]):
    on_off.iloc[i,0] = datetime.fromisoformat(on_off.iloc[i,0][:-6])  # Removing the microseconds for compatibility

on_off = on_off.drop(on_off.columns[[1,2]], axis=1)

n=0
for i in range(len(files_start)):
    delta = (files_start[i] - on_off.iloc[-1,0]).total_seconds()
    if delta < 0:
        n += 1

files = files[:n-1]


files = pd.DataFrame(np.zeros(shape=(5,len(files))), columns=files)
for i in range(n-1):
    start = files_start[i]
    e = i+1
    end = files_start[e]
    for j in range(len(on_off)):        
        condicion = (start - on_off[0][j]).total_seconds() <0 and (end - on_off[0][j]).total_seconds() >0 
        if condicion:
            files.iloc[j-(5*i), i] = j

EOD_peaks_on = []
for k in range(len(files.keys())):
    EOD = np.fromfile(files.keys()[k],dtype=np.int16) #En esta linea se carga uno de los archivos, para eso hay que indicar el indice del archivo que queremos cargar
    EOD_ch = EOD.reshape((int(EOD.shape[0]/n_channels), n_channels)) #cambiamos la forma de nuestros datos de vector a una matriz donde cada colmna es un canal

    midnight = files_start[k].replace(hour=0, minute=0, second=0, microsecond=0)
    start = abs(midnight - files_start[k]).total_seconds()
    time_EOD = np.linspace(start=start, stop=(start + EOD_ch.shape[0]/sf), num=EOD_ch.shape[0])
    time_obj = np.zeros((5*sf,5))
    for i in range(on_off.shape[0]): 
        if (files_start[k+1] - on_off.iloc[i,0]).total_seconds() > 0:
            s = abs(midnight - on_off.iloc[i,0]).total_seconds()
            time = np.linspace(s, s+5,5*sf)
            time_obj[:,i] = time 

    #Pre-procesamiento para centrar en 0 los registros de cada canal
    for i in range(n_channels):
        medianCh = np.median(EOD_ch[:,i])        
        EOD_ch[:,i] -= int(medianCh)

    # Combinamos la señal en cada canal para calcular la FB-DOE
    EOD = np.square(EOD_ch[:,0], dtype=np.int32) + np.square(EOD_ch[:,1], dtype=np.int32) #agregar termino si se tiene mas de dos canales

    # calculate the z-score 
    z_score = zscore(EOD)

        # threshold for finding EOD peaks    
    threshold = 1

    EOD_peaks, _ = find_peaks(z_score, height = threshold, distance=150)
        
    EOD_intervals = np.diff(time_EOD[EOD_peaks])
    EOD_frequencies = [1/j for j in EOD_intervals]

    for i in range(len(EOD_peaks)):
        for j in range(time_obj.shape[1]):
            range_on = [time_obj[0,j]-1, time_obj[-1,j]]
            if time_EOD[EOD_peaks[i]] > range_on[0] and time_EOD[EOD_peaks[i]] <range_on[1]:
                EOD_peaks_on.append(EOD_peaks[i])


#find jumps from one stimuli to the next
EOD_peaks_on = np.array(EOD_peaks_on)
diferencia = EOD_peaks_on[1:] - EOD_peaks_on[:-1]
saltos = []
for i in range(len(diferencia)):
    if diferencia[i] > 10000:
        saltos.append(i)
#plot
plt.figure()
x = time_EOD[EOD_peaks_on][: saltos[0]] - time_obj[0,0]
k=0
plt.scatter(x, np.ones(len(x))*k, s=.25, color='k')
k += 0.01
for j in range(1,len(saltos)):
    x = time_EOD[EOD_peaks_on][saltos[j-1]: saltos[j]] - time_obj[0,j]
    plt.scatter(x, np.ones(len(x))*k, s=.25, color='k')
    k += 0.01

plt.show()