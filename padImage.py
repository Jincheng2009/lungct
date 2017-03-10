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
label_df = pd.read_csv("../data/label_df-3mm.csv")

patients = os.listdir(datapath)
patients.sort()

z_max = np.max(label_df['z'])
xy_max = np.max(label_df['x'])
# The max dimension as the target
target_dim = [z_max,xy_max,xy_max]
print("output dimension is {}".format(target_dim))

for patient in patients:
    patient_data = np.load(datapath + patient)
    image = patient_data[0]
    sys.stdout.write("processing " + patient+ "\n")
    updated_image = add_padding(image, target_dim)
    outfile = '../data/padded/{}.npy'.format(patient)
    np.save(outfile, [updated_image, patient_data[1]])












