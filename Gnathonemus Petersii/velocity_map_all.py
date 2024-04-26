import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.spatial import distance
import pickle
from datetime import datetime
## funciones
def get_centroids(track, n_keypoints):
    centroids = []

    bodyparts = np.array([x for x in track.columns.get_level_values(1)])
    xpositions = pd.DataFrame(np.zeros((len(track),n_keypoints)), columns=np.unique(bodyparts))
    for i in range(n_keypoints):
        xpositions[np.unique(bodyparts)[i]] = (track[track.columns.get_level_values(0)[0], np.unique(bodyparts)[i], 'x'])
        
    median_xposition = np.nanmedian(xpositions, axis=1) #xpos

    ypositions = pd.DataFrame(np.zeros((len(track),n_keypoints)), columns=np.unique(bodyparts))
    for i in range(n_keypoints):
        ypositions[np.unique(bodyparts)[i]] = (track[track.columns.get_level_values(0)[0], np.unique(bodyparts)[i], 'y'])
        
    median_yposition = np.nanmedian(ypositions, axis=1) #ypos

    for j in range(len(median_xposition)):
        centroids.append([median_xposition[j], median_yposition[j]])
    return centroids

def calculate_velocity(centroids, sf, pix_to_cm):
    desplazamiento = [distance.euclidean(x,y)*pix_to_cm for x, y in zip(centroids[1:], centroids[:-1])]
    dt = len(centroids) / sf
    v = [i/dt for i in desplazamiento]

    return v

def calculate_acceleration(velocities, sf):
    dt = len(velocities) / sf
    acc = [abs(value-velocities[i-1])/dt for i, value in enumerate(velocities, start=1)]

    return acc

def plot_map(grid, objCoordinates,cmap, label,filename, vmax=None,vmin=None):
    fig, ax = plt.subplots()
    plt.imshow(grid, cmap=cmap, vmax=vmax, vmin=vmin, origin='lower')
    cbar = plt.colorbar()
    cbar.set_label(label)
    plt.show()
    #plt.scatter(objCoordinates[1]/10, objCoordinates[0]/10, s=100, c='k')
    fig.savefig(filename, format='svg', dpi=1200)

###
folder = '/Volumes/Seagate Por 7/Pose Estimation_F4'
os.chdir(folder)
files = sorted(glob.glob('*0.h5'))
print('hay ' + str(len(files)) + ' archivos')

#cargamos el archivo de FB-DOE
with open('fish4_FB-DOE.pkl', 'rb') as file:   #cambiar al nombre apropiado de archivo
        FB_doe = pickle.load(file)
files_start = [datetime.strptime(key[:-1], '%Y-%m-%dT%H_%M_%S') for key in FB_doe['FB-DOE'].keys()]

dinamics = {'velocity':{}, 'acceleration':{}}
for file in files:
    track = pd.read_hdf(file)
    track.dropna(inplace=True, how='any')
    centroids = get_centroids(track, n_keypoints=6)
    
    velocity = calculate_velocity(centroids, sf=50, pix_to_cm=12)
    vel_per_frame = pd.DataFrame(zip([round(x[0]/12) for x in centroids], [round(y[1]/12) for y in centroids], velocity), columns=['x', 'y', 'v'])
    dinamics['velocity'][file] = vel_per_frame

    acc = calculate_acceleration(velocity, sf=50)
    acc_per_frame = pd.DataFrame(zip([round(x[0]/12) for x in centroids], [round(y[1]/12) for y in centroids], acc), columns=['x', 'y', 'a'])

    dinamics['acceleration'][file] = acc_per_frame
    del track, velocity, centroids
    print('archivo' + str(file))

v_frame_all = pd.DataFrame()
for key, vel_per_frame in dinamics['velocity'].items():
    v_frame_all = pd.concat((v_frame_all, vel_per_frame))

a_frame_all = pd.DataFrame()
for key, a_per_frame in dinamics['acceleration'].items():
    a_frame_all = pd.concat((a_frame_all, a_per_frame))

dinamics_frame_all = v_frame_all
dinamics_frame_all['a'] = a_frame_all['a']
dinamics_frame_all = dinamics_frame_all.reset_index()
grouped = dinamics_frame_all.groupby(['x', 'y']).median().reset_index()

freq_vs_t = pd.DataFrame()
for key, freq in FB_doe['FB-DOE'].items():
    freq_vs_t = pd.concat((freq_vs_t, pd.DataFrame(freq)))
peak_time = pd.DataFrame()
i = 0
for key, peakT in FB_doe['Peak-time'].items():
    peak_time = pd.concat((peak_time, pd.DataFrame([p/1000+(i*60000) for p in peakT])))
    i+=1

freq_vs_t['T_vid_scale'] = peak_time 
print(dinamics_frame_all)
##fvs.t + v vs. t
fig, ax = plt.subplots()
ax1 = ax.twinx()
ax.scatter(freq_vs_t['T_vid_scale'], freq_vs_t[0], c='k')
ax1.scatter(dinamics_frame_all.index, dinamics_frame_all['v'], c='r', alpha=.5)
plt.show()

## mapas
#v_grid = np.zeros(shape=(63,62))
#a_grid = np.zeros(shape=(63,62))
#n_grid = np.zeros(shape=(63,62))

#for index, row in grouped.iterrows():
    #coords = [int(row['x']), int(row['y'])]
   # v_grid[coords[0], coords[1]] += row['v']
  #  a_grid[coords[0], coords[1]] += row['a']
 #   n_grid[coords[0], coords[1]] += 1

#v_grid = np.divide(v_grid, n_grid)
#a_grid = np.divide(a_grid, n_grid)
#plot_map(v_grid, [0,1], 'coolwarm', vmax=2, label='velocidad (cm/s)', filename='velocity_map_p6.svg')
#plot_map(a_grid, [0,1], 'coolwarm', vmax= .01, label='aceleracion (cm2/s)', filename='acceleration_map_p6.svg')

