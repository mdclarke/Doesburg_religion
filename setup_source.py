#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:47:54 2023

@author: maggie

BEFORE RUNNING
You must have:
    1) the BEM meshes created from the watershed algoritm
    2) a trans.fif file created from co-registration
    
Make sure you change all the paths in the code to save files \
    correctly on YOUR local computer
    
"""
import mne
import os.path as op

########### change this path and subject ID ##########

# define subjects_dir (where freesurfer subjects folder is) and subject number
subjects_dir = '/usr/local/freesurfer/subjects'
subject = "sub-01"

######################################################

# make boundary element model (BEM) model & solution 
# this defines the boundary of magnetic sources from the sensors
model = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.3,),
                           subjects_dir=subjects_dir)
bem_sol = mne.make_bem_solution(model)

# save bem solution file
mne.write_bem_solution(op.join(subjects_dir, "%s" %subject, 'bem', 
                               '%s-5120-bem-sol.fif' %subject), bem_sol)
# plot bem
mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir)

# read in the transformation file (-trans.fif file) 
# created during coregistration
trans = mne.read_trans(op.join(subjects_dir, "%s" %subject, 
                               '%s-trans.fif' %subject))
# read in raw MEG file
info = mne.io.read_info('/home/mdclarke/data/sub-01/sub-01_task1_raw.fif')
                        
# plot alignment (coregistration)
mne.viz.plot_alignment(
    info,
    trans,
    subject=subject,
    dig=True,
    meg=["helmet", "sensors"],
    subjects_dir=subjects_dir,
    surfaces="head-dense",
)

# set up source space - this is the grid within your boundary element model
src = mne.setup_source_space(subject, spacing="oct6", add_dist="patch", 
                             subjects_dir=subjects_dir)
print(src)
# plot src
src.plot(subjects_dir=subjects_dir)

# save your source space file
src.save(op.join(subjects_dir, "%s" %subject, 'bem', 
                               '%s-oct-6-src.fif' %subject))

# plot source space inside BEM
mne.viz.plot_bem(src=src, subject=subject, subjects_dir=subjects_dir)

### the next steps here are to create a forward model and an inverse model, 
### for these steps you will need the fully-processed MEG data
