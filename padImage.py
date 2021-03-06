# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:36:59 2017

@author: Jincheng
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 21:42:48 2017

@author: Jincheng
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
from imagelib import add_padding

# Some constants 
datapath = '../data/processed/'
label_df = pd.read_csv("../data/label_filter_df-3mm.csv")

patients = os.listdir(datapath)
patients.sort()

z_max = np.max(label_df['d1'])
xy_max = np.max(label_df['d2'])
# The max dimension as the target
target_dim = [z_max,xy_max,xy_max]
print("output dimension is {}".format(target_dim))

for patient in patients:
    patient_id = patient.split('-')[0]
    if patient_id in set(label_df['id']):
        patient_data = np.load(datapath + patient)
        image = patient_data
        sys.stdout.write("processing " + patient+ "\n")
        updated_image = add_padding(image, target_dim)
        updated_image = updated_image.astype(np.int8)
        outfile = '../data/padded/{}'.format(patient)
        np.save(outfile, updated_image)












