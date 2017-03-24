# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 21:42:48 2017

@author: Jincheng
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
from imagelib import normalize
from imagelib import load_scan
from imagelib import get_pixels_hu
from imagelib import resample
from imagelib import segment_lung_mask
from imagelib import crop_empty
import threading

# Some constants 
datapath = '../data/stage1/'
label_df = pd.read_csv("../data/stage1_labels.csv")

patients = os.listdir(datapath)
patients.sort()

label_df = pd.DataFrame({'id': patients,'mask_fraction': -1.,'d1':-1,'d2':-1,'d3':-1})

## Resolution in mm for each element in the output matrix
resolution = 3

def preprocess(patient):
    patient = row['id']
    outfile = '../data/processed/{}-{}mm.npy'.format(patient, resolution)
    sys.stdout.write(str(idx) + "\t" + patient + "\n")
    if os.path.isfile(outfile):
        return
    patient_data = load_scan(datapath + patient)
    patient_pixels = get_pixels_hu(patient_data)
    pix_resampled, spacing = resample(patient_pixels, patient_data, [resolution,resolution,resolution])
    segmented_lungs = segment_lung_mask(pix_resampled, False)
    segmented_lungs = crop_empty(segmented_lungs)
    # segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
    ## normalized_lungs = normalize(segmented_lungs)
    ###Generate metrics for QC
    mask_frac = np.sum(segmented_lungs==1) / float(np.size(segmented_lungs))
    label_df.loc[idx, 'mask_fraction'] = mask_frac
    slice_count,image_size_x,image_size_y = segmented_lungs.shape
    label_df.loc[idx, 'd1'] = slice_count
    label_df.loc[idx, 'd2'] = image_size_x
    label_df.loc[idx, 'd3'] = image_size_y
    np.save(outfile, segmented_lungs)

threads = []
for (idx, row) in label_df.iterrows():
    for i in range(4):
        patient = row['id']
        t = threading.Thread(target=preprocess, args=(patient,))
        threads.append(t)
        t.start()    
    
label_df.to_csv('../data/label_df-{}mm.csv'.format(resolution), index = False)











