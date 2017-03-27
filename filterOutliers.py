# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:38:40 2017

@author: Jincheng
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import matplotlib.pyplot as plt
import sklearn.cluster


# Some constants 
datapath = '../data/processed/'
label_df = pd.read_csv("../data/label_df-3mm.csv")

label_df["volume"] = np.log10(100+label_df["d1"] * label_df["d2"] * label_df["d3"])
plt.scatter(label_df["volume"], label_df["mask_fraction"])
## The failed segmented samples are in the left lower corner 

# Convert DataFrame to matrix
mat = label_df[["volume", "mask_fraction"]].as_matrix()
# Using sklearn
km = sklearn.cluster.KMeans(n_clusters=2)
km.fit(mat)
# Get cluster assignment labels
labels = km.labels_
# Format results as a DataFrame
results = pd.DataFrame([label_df.index,labels]).T


from sklearn import mixture
gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(mat)
labels = gmm.predict(mat)
results = pd.DataFrame([label_df.index,labels]).T

label_df["cluster"] = results[1].astype('category')
colors = {0:'red', 1:'blue'}
plt.scatter(label_df["volume"], label_df["mask_fraction"], c=label_df['cluster'].apply(lambda x : colors[x]))

## filter outliers
label_df = label_df[label_df['cluster']==0]
label_df = label_df[label_df['d1']!=177]

label_df.to_csv('../data/label_filter_df-3mm.csv', index=None)
