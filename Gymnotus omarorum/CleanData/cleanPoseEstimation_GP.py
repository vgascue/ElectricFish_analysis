# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:38:21 2023

@author: Neuropixel
"""

import pandas as pd
import numpy as np
import glob
import os

folder = r'C:\Users\Compras\Desktop\videos_DLC'

os.chdir(folder)

VIDEOfiles_names = sorted(glob.glob('*.h5'))


for vid in VIDEOfiles_names:

    track = pd.read_hdf(vid)

    likelihoods = np.zeros(shape=(len(track), 5))

    for i in range(5):

        likelihoods[:,i] = (track[track.columns.get_level_values(0)[0],track.columns.get_level_values(1).unique()[i], 'likelihood'])

    likelihoods = pd.DataFrame(likelihoods, columns=track.columns.get_level_values(1).unique())

    for index,row in likelihoods.iterrows():

        if index > 0:
            outlier = any(row < .99)
            if outlier:
                track.iloc[index, :] =[np.nan for x in track.iloc[index,:]]


    track = track.interpolate()
    name = vid[:-3] + '_clean.h5'
    track.to_hdf(name, key='track')



    

    
