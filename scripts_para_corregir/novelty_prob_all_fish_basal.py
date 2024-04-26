#Cargamos los paquetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pickle
import random
from datetime import datetime, timedelta
import matplotlib.cm as cm 
from scipy.stats import zscore

n_fish=6
#Definimos parametros
data_folder = '/Volumes/Expansion/Datos G. omarorum/' #Cambiar ruta a la carpeta donde estan los archivos .pkl (obtenidos de EOD_analysis)
os.chdir(data_folder)
folders = sorted(glob.glob('Fish*'))[1:]
p = []
ps= []
seeds=[123,4,49685, 9999,222]

def generate_timestamps(start_date):
    # Define the start and end times
        start_time = datetime(start_date.year, start_date.month, start_date.day, 21, 0, 0)
        end_time = start_time + timedelta(hours=8)  # 8 hours from 9 PM to 5 AM

        # Initialize an empty list to store timestamps
        timestamps = []

        # Generate timestamps at one-minute intervals
        current_time = start_time
        while current_time <= end_time:
            timestamps.append(current_time)
            current_time += timedelta(minutes=1)

        return timestamps


# Function to randomly sample 60 timestamps
def sample_random_timestamps(timestamps, seed):
    # Ensure that we have at least 60 timestamps available
    if len(timestamps) < 60:
        raise ValueError("Not enough timestamps to sample from.")

    random.seed(seed)
    # Sample 60 timestamps randomly
    sampled_timestamps = random.sample(timestamps, 60)
    return sampled_timestamps

for folder in folders:
    new_folder = data_folder + folder + "/Objeto/Trial 1 y 2/raw"
    os.chdir(new_folder)
    
    on_off1 = pd.read_csv(os.path.join(new_folder, 'on_off_trial1.csv'), header=None) #asegurarse de terner los .csv en la misma carpeta que el pkl
    on_off2 = pd.read_csv(os.path.join(new_folder, 'on_off_trial2.csv'), header=None)
    files_EOD = sorted(glob.glob('*.bin'))
    obj_coordinates = [291, 213]
    #definimos parametros
    sf = 10000

    #cargamos el archivo de FB-DOE
    archivo = 'fish' + folder[-1] + '_FB-DOE.pkl'
    with open(archivo, 'rb') as file:   #cambiar al nombre apropiado de archivo
            FB_doe = pickle.load(file)

    # generamos la lista files_start que contiene las timestamps en formato datetime del comienzo de cada uno de los archivos
    files_start = [datetime.strptime(key[:-1], '%Y-%m-%dT%H_%M_%S') for key in FB_doe['FB-DOE'].keys()]

    # Generamos las timestamps:
    given_date = files_start[0] 
    timestamps = generate_timestamps(given_date)
    # Sample 60 random timestamps
    for s in seeds:
        random_timestamps = sorted(sample_random_timestamps(timestamps, seed=s))
        nueve = datetime(given_date.year, given_date.month, given_date.day, 21, 0, 0)
        cinco = nueve + timedelta(hours=8) 

        keys = [x for i, x in enumerate(FB_doe['FB-DOE'].keys()) if files_start[i] > nueve and files_start[i] < cinco]
        k_idx = [i for i, x in enumerate(FB_doe['FB-DOE'].keys()) if files_start[i] > nueve and files_start[i] < cinco]
        FB_DOE = {key: FB_doe['FB-DOE'][key] for key in keys}
        Peak_time = {key: FB_doe['Peak-time'][key] for key in keys}# guardamos los peak-times en otra variable
        n_files = len(keys)
        files_start2 = [files_start[x] for x in k_idx] #nos quedamos solo con los timestamps de los archivos de interes

    #organizamos cada on segun su archivo de registro
        files = pd.DataFrame(np.zeros(shape=(20, n_files)), columns=FB_DOE.keys())  # Usamos Int64Dtype para que pueda haber nans en columnas de int

        for i, column in enumerate(files.columns[:-1], start=0):
            start = files_start2[i]
            end = files_start2[i+1]
            s = 0
            for j in range(len(random_timestamps)):
                condition = (start - random_timestamps[j]).total_seconds() < 0 and (end - random_timestamps[j]).total_seconds() > 0
                if condition:
                    files.loc[s, column] = j
                    s += 1

        files = files.dropna(how='all')
        files.replace(0, np.nan, inplace=True) # como inicializamos con una matriz de 0s, si hay algun archivo con menos de 5 ons vamos a tener 0s donde no deben haber, entonces los convertimos a nan
        files.iloc[0, 0] = 0 # el primer objeto tiene que ser un 0
        files = files.dropna(how='all')

            #inicializamos las listas
        #inicializamos las listas
        EOD_peaks_on = []
        time_EOD_all  = []
        time_obj_all = []
        EOD_f_on = []

        for k, key in enumerate(files.keys()): #loopeamos entre los archivos de interes
            midnight = files_start2[k].replace(hour=0, minute=0, second=0, microsecond=0) #definimos la media noche para el dia donde se registro ese archivo
            start = abs(midnight - files_start2[k]).total_seconds() # calculamos el tiempo de inicio del archivo en segundos totales respecto de las 00 para poder compararla
            EOD = np.fromfile(files_EOD[k],dtype=np.int16)
            time_EOD = np.linspace(start=start, stop=start+len(EOD)/sf, num=len(EOD))
            del EOD

            EOD_peaks = np.array(Peak_time[key])
            EOD_freq = np.array(FB_DOE[key])
            time_obj = np.zeros((20)) #inicializamos nuestra matriz de tiempo de prendida de obj (cada archivo puede tener maximo 5 ons, por eso las dimensiones)

            l=0
            for i in files.iloc[:,k]:
                if not np.isnan(i):
                        s = abs(midnight - random_timestamps[int(i)]).total_seconds() #calculamos el inicio del on
                        time_obj[l] = s
                        time_obj_all.append(s) #guardamos el tiempo de este on en nuestra lista de tiempos de objeto
                        time_EOD_all.append(time_EOD) #guardamos una copia de time_EOD para cada on 
                        l+=1
                
            time_peaks = time_EOD[EOD_peaks]
            EOD_zscore = zscore(EOD_freq)
            for j in range(time_obj.shape[0]):
                if not time_obj[j]==0:
                    range_on = [time_obj[j]-.5, time_obj[j]+10] #definimos el rango de interes: 1/2 segundo antes que sea el on y 2 segundos despues
                    #con = [range_on[0] <= t and t <= range_on[1] for t in videoTime]
                    condition = [range_on[0] <= time and time <= range_on[1] for time in time_peaks]

                    EOD_peaks_on.append(EOD_peaks[condition])
                    EOD_f_on.append(EOD_zscore[condition[:-1]])


        n_per_on = []
        n_peaks = []

        EOD_zscore = EOD_f_on
        if len(EOD_zscore)>0 :
            for z_score, peak, t, o in zip(EOD_zscore, EOD_peaks_on, time_EOD_all,  time_obj_all):
                peak = peak[:len(z_score)]                
                i_novel = peak[z_score > 1.8]
                novel = [i for i in z_score if i >1.8]
                if len(peak) > 0:
                    n_per_on.append(len(i_novel)/ len(peak))
                else:
                    n_per_on.append(0)
                n_peaks.append(len(peak))


        
        ps.append(sum([x*y for x,y in zip(n_per_on, n_peaks)])/630)
        p.append(sum([x*y for x,y in zip(n_per_on, n_peaks)])/sum(n_peaks))
        print(p[-1])

prob = pd.DataFrame(zip(p,ps), columns=['P does', 'P sec'])
prob.to_csv('Prob_novelty_basal.csv')