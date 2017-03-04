# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 21:42:48 2017

@author: Jincheng
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from imagelib import normalize
from imagelib import load_scan
from imagelib import get_pixels_hu
from imagelib import resample
from imagelib import segment_lung_mask

# Some constants 
datapath = '../data/samples/'
label_df = pd.read_csv("../data/stage1_labels.csv")

patients = os.listdir(datapath)
patients.sort()

label_df = label_df[label_df['id'].isin(patients)]
label_df = label_df.reset_index()
label_df['mask_fraction'] = -1.
label_df['mean_pixel'] = -1.
label_df['x'] = -1
label_df['y'] = -1
label_df['z'] = -1

processed_data = []
for (idx, row) in label_df.iterrows():
    patient = row['id']
    print(idx)
    cancer = label_df.loc[idx, 'cancer']
    patient_data = load_scan(datapath + patient)
    patient_pixels = get_pixels_hu(patient_data)
    pix_resampled, spacing = resample(patient_pixels, patient_data, [3,3,3])
    segmented_lungs = segment_lung_mask(pix_resampled, False)
    # segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
    normalized_lungs = normalize(segmented_lungs)
    processed_data.append([normalized_lungs, cancer])
    ### Generate metrics for QC
    mean_pixel = np.average(normalized_lungs)
    mask_frac = np.sum(segmented_lungs==1) / float(np.size(segmented_lungs))
    label_df.loc[idx, 'mean_pixel'] = mean_pixel
    label_df.loc[idx, 'mask_fraction'] = mask_frac
    slice_count,image_size_x,image_size_y = normalized_lungs.shape
    label_df.loc[idx, 'x'] = image_size_x
    label_df.loc[idx, 'y'] = image_size_y
    label_df.loc[idx, 'z'] = slice_count

np.save('../data/processed_data-{}-{}-{}.npy'.format(slice_count,image_size_x,image_size_y), processed_data)
label_df.to_csv('../data/label_df-{}-{}-{}.csv'.format(slice_count,image_size_x,image_size_y), index = False)











