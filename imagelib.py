# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 15:11:18 2017

@author: Jincheng
"""

import numpy as np # linear algebra
import dicom
import os
import sys
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, morphology

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
    
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)
    
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    thickness = scan[0].SliceThickness
    if thickness < 1.e-4:
    	thickness = abs(scan[1].SliceLocation - scan[2].SliceLocation)
    spacing = np.array([thickness] + scan[0].PixelSpacing, dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing
    
def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.04)
    # face_color = [0.05, 0.05, 0.15]
    mesh.set_facecolor((0.01, 0.01, 0.8, 0.04))
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

# plot_3d(pix_resampled, 400)
    
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
    
def plot_lung(datapath, fill=False):
    patient_data = load_scan(datapath)
    patient_pixels = get_pixels_hu(patient_data)
    pix_resampled, spacing = resample(patient_pixels, patient_data, [1,1,1])
    segmented_lungs = segment_lung_mask(pix_resampled, fill)
    plot_3d(segmented_lungs, 0)

## Add padding to the designed dimension
def add_padding(image, target_dim):
    z = image.shape[0]
    x = image.shape[1]
    y = image.shape[2]
    if z > target_dim[0] or x > target_dim[1] or y > target_dim[2]:
        sys.stderr.write("Input image {} is larger than the design dimenstion {}".format(image.shape, target_dim))
        return image
    pad_value = image[0][0][0]
    ## Pad the z direction
    n_upper = int((target_dim[0] - z) / 2)
    n_lower = int(target_dim[0] - z - n_upper)
    pad_upper = np.ones((n_upper,x,y)) * pad_value
    pad_lower = np.ones((n_lower,x,y)) * pad_value
    padded_image = np.append(image, pad_lower, axis=0)
    padded_image = np.append(pad_upper, padded_image, axis=0)
    ## Pad the x direction
    n_front = int((target_dim[1] - image.shape[1]) / 2)
    n_back  = int(target_dim[1] - image.shape[1] - n_front)
    pad_front = np.ones((target_dim[0],n_front,y)) * pad_value
    pad_back = np.ones((target_dim[0],n_back ,y)) * pad_value
    padded_image = np.append(padded_image, pad_back, axis=1)
    padded_image = np.append(pad_front, padded_image, axis=1)
    ## Pad the x direction
    n_left = int((target_dim[2] - image.shape[2]) / 2)
    n_right  = int(target_dim[2] - image.shape[2] - n_left)
    pad_left = np.ones((target_dim[0],target_dim[1],n_left)) * pad_value
    pad_right = np.ones((target_dim[0],target_dim[1],n_right)) * pad_value
    padded_image = np.append(padded_image, pad_right, axis=2)
    padded_image = np.append(pad_left, padded_image, axis=2)
    
    return padded_image