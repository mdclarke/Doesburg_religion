#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:26:24 2023
@author: mdclarke@sfu.ca
Use ICA to find blink and eye movement components & suppress from the data
"""
import mne
import os.path as op
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import yaml

####### set these  before running #############################################
# path to data location
path = '/home/maggie/data/sam/'
###############################################################################

# read in subject names from yaml file
with open(path / 'subjects.yaml', 'r') as fid:
    subjects = yaml.load(fid, Loader=yaml.SafeLoader)
# define filter params for EOG detection
l_freq=1
h_freq=10
for s in subjects:
    # read in processed data
    filename =  op.join(path, '%s' %s, '%s_task_all_tsss.fif' %s)
    tsss = mne.io.Raw(filename, preload=True)  
    # apply a HP filter to remove low freqeuncy drifts before performing ICA  
    # these drifts can affect the accuracy of the ICA algorithm
    filt_tsss = tsss.copy().filter(l_freq=1.0, h_freq=None)  
    # fit ICA - first 15 is probably good enough to capture blinks/heart
    ica = ICA(n_components=15, max_iter="auto", random_state=97)
    ica.fit(filt_tsss)
    ica
    del filt_tsss  
    ica.exclude = []
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(tsss, l_freq=l_freq,
                                            h_freq=h_freq)
    ica.exclude = eog_indices 
    # make some QA plots and save to disk
    # barplot of ICA components - ICs that match EOG will be shown in RED
    fig = ica.plot_scores(eog_scores)
    fig.savefig(op.join(path, '%s' %s, 'figures', 
                    '%s_ICscores.png' %s))
    plt.close(fig) 
    # plot time series of all the independant components 
    # chosen components will be shown greyed out
    # you can see the chosen components resemble blinks & eye movements
    # ica.plot_sources(tsss, show_scrollbars=False)
    # spatial patterns of ICs
    fig = ica.plot_components()
    fig.savefig(op.join(path, '%s' %s, 'figures', '%s_ICs.png' %s))
    plt.close(fig) 
    # plot data before and after removal of bad componants
    fig = ica.plot_overlay(tsss, exclude=eog_indices)
    fig.savefig(op.join(path, '%s' %s, 'figures',
                    '%s_ICAoverlay.png' %s))
    plt.close(fig) 
    # apply ICA once satisfied
    tsss_ica = ica.apply(tsss, exclude=eog_indices) 
    # save cleaned data
    tsss_ica.save(op.join(path, '%s' %s,  '%s_task_all_tsss_ica.fif' %s),
              overwrite=True)
