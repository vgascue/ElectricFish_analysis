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


def calculate_velocity(x, sf, pix_to_cm):
    desplazamiento = [abs(distance.euclidean(x,y)/pix_to_cm) for x, y in zip(x[1:], x[:-1])]
    dt = 1 / sf
    v = [i/dt for i in desplazamiento]
    return v

def plot_map(grid, objCoordinates,cmap, label,filename, vmax=None,vmin=None):
    fig, ax = plt.subplots()
    plt.imshow(grid, cmap=cmap, vmax=vmax, vmin=vmin, origin='lower')
    cbar = plt.colorbar()
    cbar.set_label(label)
    plt.scatter(objCoordinates[0]/12, objCoordinates[1]/12, s=100, c='r')
    plt.show()
    fig.savefig(filename, format='svg', dpi=1200)

###
folder = r'D:\datos_GPetersii\datos_GPetersii\Fish7\Object\Pose Estimation'
os.chdir(folder)
files = sorted(glob.glob('*.h5'))
files = files[:12]
print('hay ' + str(len(files)) + ' archivos')

dinamics = {'velocity':{}, 'acceleration':{}}
for file in files:
    track = pd.read_hdf(file)
    track.dropna(inplace=True, how='any')
    centroids = get_centroids(track, n_keypoints=6)
    
    velocity = calculate_velocity(centroids, sf=50, pix_to_cm=12)
    vel_per_frame = pd.DataFrame(zip([round(x[0]/12) for x in centroids], [round(y[1]/12) for y in centroids], velocity), columns=['x', 'y', 'v'])
    dinamics['velocity'][file] = vel_per_frame

    #acc = calculate_acceleration(velocity, sf=50)
    #acc_per_frame = pd.DataFrame(zip([round(x[0]/12) for x in centroids], [round(y[1]/12) for y in centroids], acc), columns=['x', 'y', 'a'])

    #dinamics['acceleration'][file] = acc_per_frame
    del track, velocity, centroids
    #print('archivo' + str(file))

v_frame_all = pd.DataFrame()
for key, vel_per_frame in dinamics['velocity'].items():
    v_frame_all = pd.concat((v_frame_all, vel_per_frame))

#a_frame_all = pd.DataFrame()
#for key, a_per_frame in dinamics['acceleration'].items():
 #   a_frame_all = pd.concat((a_frame_all, a_per_frame))

dinamics_frame_all = v_frame_all
#dinamics_frame_all['a'] = a_frame_all['a']
dinamics_frame_all = dinamics_frame_all.reset_index()

## mapas
v_grid = np.zeros(shape=(63,62))
#a_grid = np.zeros(shape=(63,62))
#n_grid = np.zeros(shape=(63,62))

for index, row in dinamics_frame_all.iterrows():
    coords = [int(row['x']), int(row['y'])]
    if row['v'] > 10:
        v_grid[coords[0], coords[1]] += 1
  #  a_grid[coords[0], coords[1]] += row['a']
 #   n_grid[coords[0], coords[1]] += 1

v_grid = v_grid/50
v_grid[v_grid==0] = np.nan
#a_grid = np.divide(a_grid, n_grid)
import seaborn as sns
palette = sns.color_palette("ch:s=0.9,r=-0.55", as_cmap=True)
obj_coords = np.array([492.52, 199.32])
#plot_map(v_grid, objCoordinates=obj_coords, cmap=palette, vmax=5, label='Tiempo nadando a alta velocidad (s)', filename='velocity_map_p1.svg')
#plot_map(a_grid, [0,1], 'coolwarm', vmax= .01, label='aceleracion (cm2/s)', filename='acceleration_map_p6.svg')

bins_time = []
fishdistance = np.zeros(shape=(len(v_frame_all),3))
for i in range(len(v_frame_all)):
    fishdistance[i,0]=v_frame_all.iloc[i,0]
    fishdistance[i,1]=v_frame_all.iloc[i,1]
    fishdistance[i,2]=distance.euclidean(obj_coords/12, [fishdistance[i,0],fishdistance[i,1]])

for i in np.linspace(1,30,30):
    subset = fishdistance[fishdistance[:,2]<i]
    subset = subset[subset[:,2]>i-1]
    time_high_v = []
    for j in range(len(subset)):
        time_high_v.append(v_grid[int(subset[j,0]), int(subset[j,1])])
    if len(time_high_v) > 0:
        bins_time.append(np.sum(time_high_v))

np.savetxt('bins_p7.txt', np.array(bins_time))
