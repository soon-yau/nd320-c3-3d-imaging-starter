"""
Module loads the hippocampus dataset into RAM
"""
import os
from os import listdir
from os.path import isfile, join

import numpy as np
from medpy.io import load
import nibabel as nib
from utils.utils import med_reshape

def LoadHippocampusData(root_dir, y_shape, z_shape):
    '''
    This function loads our dataset form disk into memory,
    reshaping output to common size

    Arguments:
        volume {Numpy array} -- 3D array representing the volume

    Returns:
        Array of dictionaries with data stored in seg and image fields as 
        Numpy arrays of shape [AXIAL_WIDTH, Y_SHAPE, Z_SHAPE]
    '''

    image_dir = os.path.join(root_dir, 'images')
    label_dir = os.path.join(root_dir, 'labels')

    images = [f for f in listdir(image_dir) if (
        isfile(join(image_dir, f)) and f[0] != ".")]
    max_voxel = 0
    min_voxel = 100
    out = []
    for f in images:

        # We would benefit from mmap load method here if dataset doesn't fit into memory
        # Images are loaded here using MedPy's load method. We will ignore header 
        # since we will not use it
        #image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(label_dir, f))

        image_data = nib.load(os.path.join(image_dir, f))
        image = image_data.get_fdata()

        #print(np.min(image), np.max(image))
        # TASK: normalize all images (but not labels) so that values are in [0..1] range
        bitpix = image_data.header['bitpix']
        image = (image / 2**bitpix)
        if np.max(image) > max_voxel:
            max_voxel = np.max(image)
        if np.min(image) < min_voxel:
            min_voxel = np.min(image)           
        # We need to reshape data since CNN tensors that represent minibatches
        # in our case will be stacks of slices and stacks need to be of the same size.
        # In the inference pathway we will need to crop the output to that
        # of the input image.
        # Note that since we feed individual slices to the CNN, we only need to 
        # extend 2 dimensions out of 3. We choose to extend coronal and sagittal here

        # TASK: med_reshape function is not complete. Go and fix it!
        image = med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape))
        label = med_reshape(label, new_shape=(label.shape[0], y_shape, z_shape)).astype(int)

        # TASK: Why do we need to cast label to int?
        # ANSWER: We use label as class (categorical).

        out.append({"image": image, "seg": label, "filename": f})

    # Hippocampus dataset only takes about 300 Mb RAM, so we can afford to keep it all in RAM
    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")
    print("min max", min_voxel, max_voxel)
    return np.array(out)
