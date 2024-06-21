# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:38:21 2023

@author: Neuropixel
"""
import pandas as pd
import numpy as np
import glob
import os 

def clean():

    VIDEOfiles_names = sorted(glob.glob('*.h5'))
    for vid in VIDEOfiles_names:

        track = pd.read_hdf(vid)

        likelihoods = np.zeros(shape=(len(track), 6))

        for i in range(6):

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

    
dir = r"D:\datos_GPetersii\datos_GPetersii"
os.chdir(dir)

folders = sorted(glob.glob("Fish*"))
folders = [folders[3]]

for folder in folders:
	os.chdir(dir + '\\'+folder+"\\Social\\Pose Estimation")
	videos = glob.glob("*0.h5")
	clean()
	
	print("Finished {folder}")
    
