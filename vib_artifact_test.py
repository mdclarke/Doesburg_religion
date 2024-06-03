#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:43:54 2023

@author: mdclarke@sfu.ca

"""
import mne
import os.path as op
from mne.preprocessing import maxwell_filter
import matplotlib.pyplot as plt
import yaml

####### set these  before running #############################################
path = '/home/maggie/data/sam/'
###############################################################################

# read in subject names from yaml file
with open(path / 'subjects.yaml', 'r') as fid:
    subjects = yaml.load(fid, Loader=yaml.SafeLoader)
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
  # delete old raw objects for memory
  del raw2, raw3

  # read in erm
  erm = mne.io.Raw(op.join(path, 'sam', 'empty_room.fif'),
                   preload=True)
  # apply plain tSSS fopr test purposes
  tsss = maxwell_filter(raw, calibration=fc, cross_talk=ct,
                        st_duration=10, st_correlation=0.98)
  psd = erm.plot_psd(fmax=55)
  psd.savefig(op.join(path, '%s' %i, 'figures', '%s_psd_erm' %i))
  plt.close(psd)
  psd2 = raw.plot_psd(fmax=55)
  psd2.savefig(op.join(path, '%s' %i, 'figures', '%s_psd_raw' %i))
  psd2.close(psd)
  psd3 = tsss.plot_psd(fmax=55)
  psd3.savefig(op.join(path, '%s' %i, 'figures', '%s_psd_tsss' %i))
  psd3.close(psd)
  # create proj in erm around 29.5Hz artifact
  filt_erm = erm.copy().filter(l_freq=27, h_freq=31)
  ## compute erm projectors to use for eSSS
  erm_proj = mne.compute_proj_raw(filt_erm, meg='combined')
  # plot the combined projectors
  fig = mne.viz.plot_projs_topomap(erm_proj, colorbar=True,
                                   info=erm.info)
  # save plot
  fig.savefig(op.join(path, '%s' %i, 'figures', '%s_29.5Hz_erm_proj' %i))
  plt.close(fig)
