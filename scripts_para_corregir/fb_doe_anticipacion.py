import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pickle
from datetime import datetime, timedelta
from scipy.spatial import distance

data_folder = '/Volumes/Expansion/Datos G. omarorum/Fish2/Objeto/Trial 1  y 2/raw/' #cambiar a ruta con archivos .bin
os.chdir(data_folder)
files_vid = sorted(glob.glob('*.h5'))
on_off1 = pd.read_csv("/Volumes/Expansion/Datos G. omarorum/Fish2/Objeto/Trial 1  y 2/raw/on_off_trial1.csv", header=None) #archivo csv con las timestamps del objeto
on_off2 = pd.read_csv("/Volumes/Expansion/Datos G. omarorum/Fish2/Objeto/Trial 1  y 2/raw/on_off_trial2.csv", header=None) #archivo csv con las timestamps del objeto
files_EOD = sorted(glob.glob('*.bin'))

 
with open('fish1_FB-DOE.pkl', 'rb') as file:  
        FB_doe = pickle.load(file)

files_start = []
for i in range(len(FB_doe['FB-DOE'].keys())):
    files_start.append(datetime.strptime(list(FB_doe['FB-DOE'].keys())[i][:-1], '%Y-%m-%dT%H_%M_%S'))

for i in range(on_off1.shape[0]):
    on_off1.iloc[i,0] = datetime.fromisoformat(on_off1.iloc[i,0][:-6])  # Removing the microseconds for compatibility

on_off = pd.DataFrame(np.zeros(shape=(len(on_off1), 2)), columns=['Trial 1', 'Trial 2'])
on_off['Trial 1'] = on_off1.drop(on_off1.columns[[1,2]], axis=1)

for i in range(on_off2.shape[0]):
    on_off2.iloc[i,0] = datetime.fromisoformat(on_off2.iloc[i,0][:-6])  # Removing the microseconds for compatibility

on_off['Trial 2'] = on_off2.drop(on_off2.columns[[1,2]], axis=1)

#elegimos los indices de archivos donde se dan los trials (T1 tarde, T2 mañana)
n_trial1=[]
n_trial2=[]
for i in range(len(files_start)):
    delta1 = (files_start[i] - on_off['Trial 1'].iloc[-1]).total_seconds()
    delta2 = (files_start[i] - on_off['Trial 2'].iloc[-1]).total_seconds()
    delta3 = (files_start[i] - on_off['Trial 2'].iloc[0]).total_seconds()
    if delta1 < 0:
        n_trial1.append(i)
    if delta2 < 0 and delta3 > 0:
        n_trial2.append(i)

on_off= pd.DataFrame(on_off.values.flatten(), columns=['On_off'])

files_start = [files_start[x] for x in n_trial1+n_trial2]
FB_DOE = FB_doe['FB-DOE']
Peak_time = FB_doe['Peak-time']

keys1 = [list(FB_doe['FB-DOE'].keys())[i] for i in n_trial1]
keys2 = [list(FB_doe['FB-DOE'].keys())[i] for i in n_trial2]

FB_DOE_T1 = {key: FB_DOE.get(key) for key in keys1}
Peaks_T1 = {key: Peak_time.get(key) for key in keys1}
FB_DOE_T2 = {key: FB_DOE.get(key) for key in keys2}
Peaks_T2 = {key: Peak_time.get(key) for key in keys2}

n_files = len(FB_DOE_T1.keys())+len(FB_DOE_T2.keys())

#organizamos cada on segun su archivo de registro
files = pd.DataFrame(np.zeros(shape=(10,n_files)), columns=list(FB_DOE_T1.keys())+list(FB_DOE_T2.keys()))
for i in range(1,len(files.columns)):
    start = files_start[i-1]
    end = files_start[i]
    s=0 
    for j in range(len(on_off)):            
        condicion = (start - on_off.iloc[j,0]).total_seconds() < 0 and (end - on_off.iloc[j,0]).total_seconds() > 0 
        if condicion:
            files.iloc[s, i-1] = j
            s+=1


files.replace(0, np.nan, inplace=True) # como inicializamos con una matriz de 0s, si hay algun archivo con menos de 5 ons vamos a tener 0s donde no deben haber, entonces los convertimos a nan
files = files.dropna(how='all')
files.iloc[0,0] = 0 # el primer objeto tiene que ser un 0
for col in list(files.keys()): #este codigo es para asegurarnos que hayan quedado los numeros de on en orden
    files[col] = sorted(files[col])
print(files)
#inicializamos las listas
EOD_peaks_on = []
time_EOD_all  = []
time_obj_all = []
EOD_f_on = []

for k in range(len(files.keys())): #loopeamos entre los archivos de interes
    midnight = files_start[k].replace(hour=0, minute=0, second=0, microsecond=0) #definimos la media noche para el dia donde se registro ese archivo
    start = abs(midnight - files_start[k]).total_seconds() # calculamos el tiempo de inicio del archivo en segundos totales respecto de las 00 para poder compararla
    EOD = np.fromfile(files_EOD[k],dtype=np.int16)
    time_EOD = np.linspace(start=start, stop=start+(20*60), num=len(EOD))
    del EOD
    EOD_peaks = Peak_time[list(Peak_time.keys())[k]]
    EOD_freq = np.array(FB_DOE[list(FB_DOE.keys())[k]])
    time_obj = np.zeros((1,10)) #inicializamos nuestra matriz de tiempo de prendida de obj (cada archivo puede tener maximo 5 ons, por eso las dimensiones)
    
    l=0
    for i in files.iloc[:,k]: 
           if not np.isnan(i):
                s = abs(midnight - on_off.iloc[int(i),0]).total_seconds() #calculamos el inicio del on
                time_obj[0,l] = s
                time_obj_all.append(s) #guardamos el tiempo de este on en nuestra lista de tiempos de objeto
                time_EOD_all.append(time_EOD) #guardamos una copia de time_EOD para cada on 
                l+=1

    for j in range(time_obj.shape[1]):
        range_on = [time_obj[0,j]-0.5, time_obj[0,j]+10] #definimos el rango de interes: 1/2 segundo antes que sea el on y 2 segundos despues
        
        time_peaks = time_EOD[EOD_peaks]
        condition = [range_on[0] <= time <= range_on[1] for time in time_peaks]

        EOD_peaks_on.append(EOD_peaks[condition])
        EOD_f_on.append(EOD_freq[condition[:-1]])
  

    print('termino archivo ' + str(k))

import matplotlib.cm as cm
colormap = cm.get_cmap('Blues')
colors = [colormap(i+100) for i in range(len(time_obj_all))]

EOD_peaks_on = [sublist for sublist in EOD_peaks_on if len(sublist) > 0]
EOD_f_on = [sublist for sublist in EOD_f_on if len(sublist) > 0]


EOD_peaks_on1 = EOD_peaks_on[:len(on_off1)]
EOD_peaks_on2 = EOD_peaks_on[len(on_off1):]

EOD_f_on1 = EOD_f_on[:len(on_off1)]
EOD_f_on2 = EOD_f_on[len(on_off1):]

time_EOD_all1 = time_EOD_all[:len(on_off1)]
time_EOD_all2 = time_EOD_all[len(on_off1):]

time_obj_all1 = time_obj_all[:len(on_off1)]
time_obj_all2 = time_obj_all[len(on_off1):]


num_figures = 6
num_plots_per_figure = 10

plt.figure()
n=0
for freq1, freq2, peak1, peak2, time1, time2, obj1, obj2, color in zip(EOD_f_on1, EOD_f_on2, EOD_peaks_on1, EOD_peaks_on2, time_EOD_all1, time_EOD_all2, time_obj_all1, time_obj_all2, colors):
    x1 = time1[peak1] - obj1
    x2 = time2[peak2] - obj2
    y1 = freq1
    y2 = freq2
    n += 1
    if n > 10:
        n = 0 
        plt.axvline(x=0, color='red', linestyle='-', label='Vertical Line')
        plt.axvline(x=5, color='red', linestyle='-', label='Vertical Line')
        #plt.plot(np.linspace(-.5, 10, len(medians)), medians, color='k')
        plt.show()
        plt.pause(0.1)
        plt.figure()
    if len(x1) == len(y1):
        plt.scatter(x1, y1, s=2.85, color=color)
        plt.plot(x1, y1, color=color)
    if len(x2) == len(y2):
        plt.scatter(x2, y2, s=2.85, color=color)
        plt.plot(x2, y2, color=color)


