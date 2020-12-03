#!/usr/bin/env python
# coding: utf-8
"""
Usage: python tools/nifti2slices.py --data_path data/MyDataset/NiftiParentFolder
"""
import argparse
import numpy as np
import os
import nibabel as nib
from tqdm import tqdm


def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header
    :param img_path: (string) Path of the 'nii' or 'nii.gz' image file name
    :return: Three element, the first is a numpy array of the image values (height, width, slices, phases),
             ## (No) the second is the affine transformation of the image, and the
             ## (No) last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return np.asanyarray(nimg.dataobj), nimg.affine, nimg.header


def parse_args():
    parser = argparse.ArgumentParser(description='Convert your nifti volume dataset (3D) to numpy slices (2D)!')
    parser.add_argument('--data_path', type=str, required=True, help='Parent folder with nifti files.')
    aux = parser.parse_args()
    arguments = aux.data_path
    return arguments


data_path = parse_args()
print("Running...")
all_nifti_paths = []
for subdir, dirs, files in os.walk(data_path):
    for file in files:
        entry = os.path.join(subdir, file)
        if entry.endswith((".nii", ".nii.gz")):
            all_nifti_paths.append(entry)

for nifit_path in tqdm(all_nifti_paths, desc="Remaining Files"):
    nifti_volume = load_nii(nifit_path)[0]
    dims = nifti_volume.shape  # h, w, slices, *phases*
    extension_length = 4 if nifit_path.endswith(".nii") else 7
    if len(dims) == 3:  # not phases, volume for specific phase
        for s in range(dims[2]):  # slices
            current_slice = nifti_volume[..., s]
            current_slice_path = f"{nifit_path[:-extension_length]}_slice{s}.npy"
            np.save(current_slice_path, current_slice)
    else:
        for s in range(dims[2]):  # slices
            for p in range(dims[3]):  # phases
                current_slice = nifti_volume[..., s, p]
                current_slice_path = f"{nifit_path[:-extension_length]}_slice{s}_phase{p}.npy"
                np.save(current_slice_path, current_slice)
