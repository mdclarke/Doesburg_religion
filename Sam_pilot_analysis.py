#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:32:52 2023

@author: mdclarke
"""
import mne
import os.path as op
from mne.preprocessing import maxwell_filter
from mne.chpi import compute_chpi_amplitudes, compute_chpi_locs, compute_head_pos

path = '/media/mdclarke/a19e7abc-95df-4509-855d-e868f0e34f0f/sam/'

# read in system specific cross-talk and fine calibration files
ct = op.join(path, 'ct_sparse.fif')
fc = op.join(path, 'sss_cal.dat')

# read in raw data
raw = mne.io.Raw(op.join(path, 'pilot_4_task_1_raw.fif'), preload=True)

# make a copy of raw to process
raw2 = raw.copy()

# bandpass filter 
# (I chose 1-40 Hz but you can change this to what you like or skip it)
raw2.filter(l_freq=1, h_freq=40)

# plot raw data and visually inspect for bad channels
raw2.plot()

# set bad channels in data structure (you can edit this, it's not final)
raw2.info['bads'] = ['MEG2122', 'MEG1611']

# filter out the cHPI frequencies - only use this if you don't filter.
# the lowpass filter will get rid of HPI high frequencies.I commented this step
# out for now.
#mne.chpi.filter_chpi(raw)

# setup for movement compensation by extracting coil info
# (perform on unfiltered raw)
amps = compute_chpi_amplitudes(raw)
locs = compute_chpi_locs(raw.info, amps)
pos = compute_head_pos(raw.info, locs)
# plot
mne.viz.plot_head_positions(pos, mode='traces')

# perform tSSS with default params
tsss = maxwell_filter(raw2, calibration=fc, cross_talk=ct, st_duration=10,
                     head_pos=pos, destination=(0.,0.,0.04))

# find triggers
events=mne.find_events(tsss, stim_channel='STI101', 
                       shortest_event=1/raw.info['sfreq'])
# plot triggers
mne.viz.plot_events(events)







