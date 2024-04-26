###
#Este script calcula las likelihoods por keypoint para un conjunto de videos (por ejemplo, por cada pez). Es la funcion extendida de 'DLC_explore' para muchos videos. 
#La salida del script es un histograma de likelihood para todos los videos. Este histograma se guarda en formato svg en la misma carpeta donde estan los archivos h5
#antes de correr hay que cambiar la ruta a la carpeta con los archivos, asi como el parametro n_keypoints segun el numero de keypoints trackeados. 
###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import matplotlib.cm as cm

folder = '/Volumes/Expansion/datos_GPetersii/datos_GPetersii/Fish1/Social/Pose estimation' #seleccionar la carpeta con los archivos h5
os.chdir(folder)
files = sorted(glob.glob('*.h5'))
n_keypoints = 6 # 6 para petersii, 5 para omarorum

likelihoods_all = np.zeros(shape=(1,n_keypoints))
for i in range(len(files)):
    track = pd.read_hdf(files[i])
    likelihoods = np.zeros(shape=(len(track), n_keypoints))

    for j in range(n_keypoints):
        likelihoods[:,j] = (track[track.columns.get_level_values(0)[0],track.columns.get_level_values(1).unique()[j], 'likelihood'])

    likelihoods_all = np.concatenate((likelihoods_all, likelihoods), axis=0)
    print('termino archivo ' + str(i))


likelihoods_all = pd.DataFrame(likelihoods_all, columns=track.columns.get_level_values(1).unique())

likelihoods_all.head()
print(likelihoods_all.shape)

colormap = cm.get_cmap('viridis')
colors = [colormap(i) for i in range(n_keypoints)]

ax, fig = plt.subplots()
for col in likelihoods_all.columns:
    plt.hist(likelihoods_all[col], bins= 20, alpha=.6, label=col, density=True)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()
plt.show()
fig.savefig('DLC_likelihood_f1.png', dpi=1200)