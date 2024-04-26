#Cargamos los paquetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pickle
from datetime import datetime, timedelta
import matplotlib.cm as cm 
from scipy.stats import zscore

n_fish=6
#Definimos parametros
data_folder = '/Volumes/Expansion/Datos G. omarorum/' #Cambiar ruta a la carpeta donde estan los archivos .pkl (obtenidos de EOD_analysis)
os.chdir(data_folder)
folders = sorted(glob.glob('*'))[2:]
p = []
ps= []
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

    on_off1[0] = on_off1[0].apply(lambda x: datetime.fromisoformat(x[:-6]))

# ahora pasamos on_off1 a formato dataframe y nos quedamos solo con la columna de tiempo de on 
    on_off = pd.DataFrame(np.zeros(shape=(len(on_off1), 2)), columns=['Trial 1', 'Trial 2'])
    on_off['Trial 1'] = on_off1.drop(on_off1.columns[[1, 2]], axis=1)

# hacemos lo mismo con on_off2 y lo agregamos al dataframe on_off
    on_off2[0] = on_off2[0].apply(lambda x: datetime.fromisoformat(x[:-6]))
    on_off['Trial 2'] = on_off2.drop(on_off2.columns[[1, 2]], axis=1)

    # guardamos algunos tiempos de referencia para filtrar los archivos 
    trial1_last = on_off['Trial 1'].iloc[-1]
    trial2_last = on_off['Trial 2'].iloc[-1]
    trial2_first = on_off['Trial 2'].iloc[0]

    # generamos n_trial1 y n_trial2 que contiene los indices de los archivos que nos interesan para cada trial
    n_trial1 = [i for i, start in enumerate(files_start) if (start - trial1_last).total_seconds() < 0]
    n_trial2 = [i for i, start in enumerate(files_start) if (start - trial2_last).total_seconds() < 0 and (start - trial2_first).total_seconds() > 0]

    # Achatamos on_off y lo ordenamos
    on_off = pd.DataFrame(on_off.values.flatten(), columns=['On_off']).sort_values(by='On_off')
    on_off.reset_index(inplace=True)

    files_start = [files_start[x] for x in n_trial1 + n_trial2] #nos quedamos solo con los timestamps de los archivos de interes

    FB_DOE = FB_doe['FB-DOE'] #guardamos las frequencias en una variable
    Peak_time = FB_doe['Peak-time'] # guardamos los peak-times en otra variable

    keys1 = [list(FB_DOE.keys())[i] for i in n_trial1] #guardamos las keys del trial 1
    keys2 = [list(FB_DOE.keys())[i] for i in n_trial2] #guardamos las keys del trial 2

    FB_DOE_T1 = {key: FB_DOE.get(key) for key in keys1} # guardamos las frecuencias del trial 1 
    Peaks_T1 = {key: Peak_time.get(key) for key in keys1}  #guardamos los peak-times del trial 1
    FB_DOE_T2 = {key: FB_DOE.get(key) for key in keys2}  #guardamos las frecuencias del trial 2
    Peaks_T2 = {key: Peak_time.get(key) for key in keys2} #guardamos los peak-times del trial 2

    n_files = len(FB_DOE_T1) + len(FB_DOE_T2) #guardamos el numero de files con el que estamos trabajando para usar despues

    #organizamos cada on segun su archivo de registro
    files = pd.DataFrame(np.zeros(shape=(4, n_files)), columns=list(FB_DOE_T1.keys()) + list(FB_DOE_T2.keys()))  # Usamos Int64Dtype para que pueda haber nans en columnas de int

    for i, column in enumerate(files.columns[:-1], start=0):
        start = files_start[i]
        end = files_start[i+1]
        s = 0
        for j in range(len(on_off)):
            condition = (start - on_off['On_off'][j]).total_seconds() < 0 and (end - on_off['On_off'][j]).total_seconds() > 0
            if condition:
                files.loc[s, column] = j
                s += 1

    files = files.dropna(how='all')
    files.replace(0, np.nan, inplace=True) # como inicializamos con una matriz de 0s, si hay algun archivo con menos de 5 ons vamos a tener 0s donde no deben haber, entonces los convertimos a nan
    files.iloc[0, 0] = 0 # el primer objeto tiene que ser un 0
    files = files.dropna(how='all')

        #inicializamos las listas
    EOD_peaks_on = {'Trial 1': [], 'Trial 2': []}
    time_EOD_all  = {'Trial 1': [], 'Trial 2': []}
    time_obj_all = {'Trial 1': [], 'Trial 2': []}
    EOD_f_on = {'Trial 1': [], 'Trial 2': []}
    distancia = {'Trial 1': [], 'Trial 2': [], 'Trial 2T' : [], 'Trial 1T': []}

    for k in range(len(files.keys())-1): #loopeamos entre los archivos de interes
        midnight = files_start[k].replace(hour=0, minute=0, second=0, microsecond=0) #definimos la media noche para el dia donde se registro ese archivo
        start = abs(midnight - files_start[k]).total_seconds() # calculamos el tiempo de inicio del archivo en segundos totales respecto de las 00 para poder compararla
        EOD = np.fromfile(files_EOD[k],dtype=np.int16)
        time_EOD = np.linspace(start=start, stop=start+len(EOD)/sf, num=len(EOD))
        del EOD

        time_obj = np.zeros((6)) #inicializamos nuestra matriz de tiempo de prendida de obj (cada archivo puede tener maximo 5 ons, por eso las dimensiones)
        
        if k < len(n_trial1):
            key = 'Trial 1'
            EOD_peaks = np.array(Peaks_T1[list(Peaks_T1.keys())[k]])
            EOD_freq = np.array(FB_DOE_T1[list(FB_DOE_T1.keys())[k]])
        
        else:
            key = 'Trial 2'
            EOD_peaks = np.array(Peaks_T2[list(Peaks_T2.keys())[k-len(n_trial1)]])
            EOD_freq = np.array(FB_DOE_T2[list(FB_DOE_T2.keys())[k-len(n_trial1)]])
            
        l=0
        for i in files.iloc[:,k]:
            if not np.isnan(i):
                    s = abs(midnight - on_off['On_off'][int(i)]).total_seconds() #calculamos el inicio del on
                    time_obj[l] = s
                    time_obj_all[key].append(s) #guardamos el tiempo de este on en nuestra lista de tiempos de objeto
                    time_EOD_all[key].append(time_EOD) #guardamos una copia de time_EOD para cada on 
                    l+=1
        
        time_peaks = time_EOD[EOD_peaks]
        EOD_zscore = zscore(EOD_freq)
        for j in range(time_obj.shape[0]):
            if not time_obj[j]==0:
                range_on = [time_obj[j]-.5, time_obj[j]+10] #definimos el rango de interes: 1/2 segundo antes que sea el on y 2 segundos despues
                #con = [range_on[0] <= t and t <= range_on[1] for t in videoTime]
                condition = [range_on[0] <= time and time <= range_on[1] for time in time_peaks]

                EOD_peaks_on[key].append(EOD_peaks[condition])
                EOD_f_on[key].append(EOD_zscore[condition[:-1]])

    colormap = cm.get_cmap('cool')
    colors1 = [colormap(i+100) for i in range(len(time_obj_all['Trial 1']))]
    colormap = cm.get_cmap('cool')
    colors2 = [colormap(i+100) for i in range(len(time_obj_all['Trial 2']))]
    colors = {'Trial 1': colors1, 'Trial 2':  colors2}

    n_per_on = []
    t_novelty = []
    trials = []
    n_peaks = []
    day_switch = len(EOD_peaks_on['Trial 1'])

    k=0
    for key in EOD_peaks_on.keys():
        time = time_EOD_all[key]
        obj = time_obj_all[key]
        EOD_zscore = EOD_f_on[key]
        peaks = EOD_peaks_on[key]

        if len(EOD_zscore)>0 :
            for z_score, peak, t, o, c in zip(EOD_zscore, peaks, time,  obj, colors[key]):
                peak = peak[:len(z_score)]                
                i_novel = peak[z_score > 1.8]
                novel = [i for i in z_score if i >1.8]
                if len(peak) > 0 :
                    n_per_on.append(len(i_novel)/len(peak))
                else:
                    n_per_on.append(0)
                n_peaks.append(len(peak))
                t_novelty.append(o)
                trials.append(k)
                k += .01
    
    ps.append(sum([x*y for x,y in zip(n_per_on, n_peaks)])/630)
    p.append(sum([x*y for x,y in zip(n_per_on, n_peaks)])/sum(n_peaks))
    print(p[-1])

prob = pd.DataFrame(zip(p,ps), columns=['P does', 'P sec'])
prob.to_csv('Prob_novelty_dur_obj.csv')