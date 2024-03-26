#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 09:20:23 2023

@author: mdclarke@sfu.ca

Append task data files, identify bad channels, apply a low-pass filter,
perform tSSS & movement compensation
"""
import mne
import os
import os.path as op
from mne.preprocessing import maxwell_filter, compute_average_dev_head_t, find_bad_channels_maxwell
from mne.chpi import (compute_chpi_amplitudes, compute_chpi_locs, 
                      compute_head_pos, write_head_pos)

####### set these  before running #############################################
path = '/home/maggie/data/sam/'

# make a list of all subject names
subjects = ['subject_1', 'subject_2', 'subject_3', 'subject_4',
            'subject_5', 'subject_6', 'subject_7', 'subject_8']

# set to True if headpos has not already been calculated & saved
calc_headpos = True
###############################################################################

# read in system specific cross-talk and fine calibration files 
ct = op.join(path, 'ct_sparse.fif')
fc = op.join(path, 'sss_cal.dat')

for i in subjects:
  # append task runs & save output
  raw = mne.io.Raw(op.join(path, '%s' %i, 
                           '%s_task_run1_raw.fif' %i), preload=True)
  raw2 = mne.io.Raw(op.join(path, '%s' %i, 
                           '%s_task_run2_raw.fif' %i), preload=True)
  raw3 = mne.io.Raw(op.join(path, '%s' %i, 
                           '%s_task_run3_raw.fif' %i), preload=True)
  total_times = raw.times.max() + raw2.times.max() + raw3.times.max()

  raw.append([raw2, raw3])
  assert round(raw.times.max(), 2) == round(total_times, 2)

  # save new combined raw file
  raw.save(op.join(path, '%s' %i, '%s_task_all_raw.fif' %i))
  # delete old raw objects for memory
  del raw, raw2, raw3

  # read in raw data
  raw = mne.io.Raw(op.join(path, '%s' %i, 
                           '%s_task_all_raw.fif' %i), preload=True)
  
  # automatically detect bad channels
  raw_check = raw.copy()
  noisy_chs, flat_chs, scores = find_bad_channels_maxwell(
    raw_check,
    cross_talk=ct,
    calibration=fc,
    return_scores=True,
    verbose=True,
    )
  print("Noisy channels: ", noisy_chs)
  print("Flat channels: ", flat_chs)
  
  # make a copy of raw and apply a lowpass filter
  raw_filtered = raw.copy().load_data().filter(l_freq=None, h_freq=55.)

  # set bad channels in data structure - get these from 
  # runsheet OR visual inspection
  raw_filtered.info['bads'] = noisy_chs + flat_chs

  # setup for movement compensation by extracting coil info
  # (perform on unfiltered raw)
  if calc_headpos == True:
      amps = compute_chpi_amplitudes(raw)
      locs = compute_chpi_locs(raw.info, amps)
      pos = compute_head_pos(raw.info, locs)
      # save pos file (computing takes awhile)
      write_head_pos(op.join(path, '%s' %i, '%s_task_all_pos.fif' %i), pos)
  else:
      pos = mne.chpi.read_head_pos(op.join(path, '%s' %i, 
                                           '%s_task_all_pos.fif' %i))

  # get initial head position and calculate average position (across time)
  orig_head_dev_t = mne.transforms.invert_transform(raw.info["dev_head_t"])
  avg_head_dev_t = mne.transforms.invert_transform(compute_average_dev_head_t
                                                       (raw, pos))

  # plot head positions over time (green=initial, red=average)
  if not os.path.exists(op.join(path, '%s' %i, 'figures')):
      os.makedirs(op.join(path, '%s' %i, 'figures'))

  fig = mne.viz.plot_head_positions(pos)
  for ax, val, val_ori in zip(
      fig.axes[::2],
      avg_head_dev_t["trans"][:3, 3],
      orig_head_dev_t["trans"][:3, 3],
  ):
      ax.axhline(1000 * val, color="r")
      ax.axhline(1000 * val_ori, color="g")
  fig.savefig(op.join(path, '%s' %i, 'figures', '%s_task_all_headpos' %i))

  # use average position for movement compensation destination    
  destination = (raw.info['dev_head_t']['trans'][:3, 3])

  # perform tSSS with movement compensation 
  # use higher CL (0.99) because of filtering
  tsss_mc = maxwell_filter(raw_filtered, calibration=fc, cross_talk=ct,
                           st_duration=10, head_pos=pos, 
                           destination=destination, st_correlation=0.99)

  tsss_mc.save(op.join(path, '%s' %i, '%s_task_all_tsss.fif' %i))
