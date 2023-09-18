#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:38:23 2023

@author: maggie

Resample data, do epoching and save averaged file
"""
import mne
import os.path as op
import matplotlib.pyplot as plt
import numpy as np

####### set these  before running #############################################
path = '/home/maggie/data/sam/'

# setup subject name ### change this for each subject
s = 'pilot_6'
###############################################################################

# read in processed data
filename =  op.join(path, '%s' %s, '%s_task_all_tsss_ica.fif' %s)
tsss = mne.io.Raw(filename, preload=True)

# find triggers
events = mne.find_events(tsss, stim_channel='STI101', 
                     shortest_event=1/tsss.info['sfreq'])

# make copy of events 
adjusted_events = events.copy()

# adjust triggers for visual delay
t_adjust= -80e-3 #80 ms
t_adj = int(np.round(t_adjust *tsss.info['sfreq']))
adjusted_events[:,0] += t_adj

# define trigger of interest
event_dict = {
  "human": 131,
  "inan": 121,
  "god": 111,
  "super":141}

# plot triggers (events)
fig = mne.viz.plot_events(
adjusted_events, event_id=event_dict, sfreq=tsss.info["sfreq"], 
first_samp=tsss.first_samp)
fig.savefig(op.join(path, '%s' %s, 'figures', '%s_events' %s))
plt.close(fig)

#reject mags and grads > 9000 fT, 4000 fT/cm
reject = dict(mag=9000e-15, grad=4000e-13)

# epoch data w respect to event onset
epochs = mne.Epochs(tsss, adjusted_events, event_id=event_dict,
                    tmin=-0.1, tmax=0.7, baseline=(-0.1, 0), 
                    reject=reject, preload=True)
# plot epochs
#epochs.plot(events=events)

# plot drop log - this is how many epochs got dropped from the 
# amplitude rejection
fig = epochs.plot_drop_log()
fig.savefig(op.join(path, '%s' %s, 'figures', '%s_droplog' %s))
plt.close(fig)

ev = epochs.average(by_event_type=True)
mne.write_evokeds(op.join(path, '%s' %s, '%s_att_tsss_ica_ave.fif' %s),
                  ev, overwrite=True)

# average epoched data
human = epochs["human"].average()
inan = epochs["inan"].average()
god = epochs["god"].average()
super = epochs["super"].average()

# plot average
human.plot_joint()
inan.plot_joint()
god.plot_joint()
super.plot_joint()
