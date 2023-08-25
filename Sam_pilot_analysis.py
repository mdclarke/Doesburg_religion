#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 09:20:23 2023

@author: maggie
"""
import mne
import os
import os.path as op
from mne.preprocessing import maxwell_filter, compute_average_dev_head_t
from mne.chpi import compute_chpi_amplitudes, compute_chpi_locs, compute_head_pos, write_head_pos

####### set these  before running #############################################
path = '/Applications/freesurfer/7.4.1/subjects/pilot_data/'

# setup subject name ### change this for each subject
s = 'sub-01'
c = 'task_pilot-1'

# set to True if headpos has not already been calculated & saved
calc_headpos = True
###############################################################################

# read in system specific cross-talk and fine calibration files 
# (make sure you place these in the folder listed as 'path' above)
ct = op.join(path, 'ct_sparse.fif')
fc = op.join(path, 'sss_cal.dat')

# read in raw data
raw = mne.io.Raw(op.join(path, '%s' %s, 'meg',
                         '%s.fif' %c), preload=True)

# make a copy of raw and filter (you can change the bandpass here or choose 
# to not filter) - right now this is set to lowpass at 55 Hz
raw_filtered = raw.copy().load_data().filter(l_freq=None, h_freq=55.)

# plot raw data and visually inspect for bad channels
raw_filtered.plot()

# set bad channels in data structure - get these from 
# runsheet OR visual inspection
raw_filtered.info['bads'] = ['MEG1611', 'MEG1213']

## filter out the cHPI frequencies - only use this if you don't bandpass filter.
## the lowpass filter will get rid of HPI high frequencies. I commented this 
## step out for now.
#mne.chpi.filter_chpi(raw)

# setup for movement compensation by extracting coil info
# (perform on unfiltered raw)
if calc_headpos == True:
    amps = compute_chpi_amplitudes(raw)
    locs = compute_chpi_locs(raw.info, amps)
    pos = compute_head_pos(raw.info, locs)
    # save pos file (computing takes awhile)
    write_head_pos(op.join(path, '%s' %s, 'meg', '%s_pos.fif' %c), pos)
else:
    pos = mne.chpi.read_head_pos(op.join(path, '%s' %s, 'meg',
                                         '%s_pos.fif' %c))

# get initial head position and calculate average (across time) head position
orig_head_dev_t = mne.transforms.invert_transform(raw.info["dev_head_t"])
avg_head_dev_t = mne.transforms.invert_transform(compute_average_dev_head_t
                                                     (raw, pos))

# plot head positions over time (green=initial, red=average)
if not os.path.exists(op.join(path, '%s' %s, 'meg', 'figures')):
    os.makedirs(op.join(path, '%s' %s, 'meg', 'figures'))
write_head_pos(op.join(path, '%s' %s, 'meg', 'figures', '%s_pos.fif' %c), pos)

fig = mne.viz.plot_head_positions(pos)
for ax, val, val_ori in zip(
    fig.axes[::2],
    avg_head_dev_t["trans"][:3, 3],
    orig_head_dev_t["trans"][:3, 3],
):
    ax.axhline(1000 * val, color="r")
    ax.axhline(1000 * val_ori, color="g")
fig.savefig(op.join(path, '%s' %s, 'meg', 'figures', '%s_headpos' %c))

# use average position for movement compensation destination    
destination = (raw.info['dev_head_t']['trans'][:3, 3])

# perform tSSS - some params for this function will change when running
# pediatric subjects
tsss_mc = maxwell_filter(raw_filtered, calibration=fc, cross_talk=ct,
                         st_duration=10, head_pos=pos, 
                         destination=destination)
tsss_mc.save(op.join(path, '%s' %s, 'meg', '%s_tsss.fif' %c))

# find triggers
events = mne.find_events(raw, stim_channel='STI101', 
                         shortest_event=1/raw.info['sfreq'])
# plot triggers
fig = mne.viz.plot_events(
    events, sfreq=raw.info["sfreq"], 
    first_samp=raw.first_samp)
fig.savefig(op.join(path, '%s' %s, 'meg', 'figures', '%s_events' %c))
