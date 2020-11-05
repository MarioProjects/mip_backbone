import nibabel as nib
import albumentations
import numpy as np
import os
import pandas as pd
import torch
import copy
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader
from utils.data_augmentation import common_test_augmentation
import pydicom
from PIL import Image
from utils.datasets import *


def load_tif(tif_path):
    return np.array(Image.open(tif_path))


def read_dicom(dcm_path):
    f = pydicom.read_file(dcm_path)
    img = f.pixel_array.astype('int16')
    return img


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


def save_nii(img_path, data, affine, header):
    """
    Save a nifty file
    :param img_path: Path to save image file name
    :param data: numpy array of the image values
    :param affine: nii affine transformation of the image
    :param header: nii header of the image
    :return:(void)
    """
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


def add_depth_channels(image_tensor):
    _, h, w = image_tensor.size()
    image = torch.zeros([3, h, w])
    image[0] = image_tensor
    for row, const in enumerate(np.linspace(0, 1, h)):
        image[1, row, :] = const
    image[2] = image[0] * image[1]
    return image


def apply_normalization(image, normalization_type):
    """
    https://www.statisticshowto.com/normalized/
    :param image: numpy image
    :param normalization_type: one of defined normalizations
    :return: normalized image
    """
    if normalization_type == "none":
        return image
    elif normalization_type == "reescale":
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min)
        return image
    elif normalization_type == "standardize":
        mean = np.mean(image)
        std = np.std(image)
        image = image - mean
        image = image / std
        return image
    assert False, "Unknown normalization: '{}'".format(normalization_type)


def apply_volume_normalization(list_images, normalization_type):
    for indx, image in enumerate(list_images):
        list_images[indx, ...] = apply_normalization(image, normalization_type)
    return list_images


def apply_torch_normalization(image, normalization_type):
    """
    https://www.statisticshowto.com/normalized/
    :param image: pytorch image
    :param normalization_type: one of defined normalizations
    :return: normalized image
    """
    if normalization_type == "none":
        return image
    elif normalization_type == "reescale":
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min)
        return image
    elif normalization_type == "standardize":
        mean = image.mean().detach()
        std = image.std().detach()
        image = image - mean
        image = image / (std + 1e-10)
        return image
    assert False, "Unknown normalization: '{}'".format(normalization_type)


def apply_augmentations(image, transform, img_transform, mask=None):
    if transform:
        if mask is not None:
            augmented = transform(image=image, mask=mask)
            mask = augmented['mask']
        else:
            augmented = transform(image=image)

        image = augmented['image']

    if img_transform:
        augmented = img_transform(image=image)
        image = augmented['image']

    return image, mask


def apply_volume_augmentations(list_images, transform, img_transform):
    """
    Apply same augmentations to volume images
    :param list_images: (array) [num_images, height, width] Images to transform
    :return: (array) [num_images, height, width] Transformed Images
    """
    if img_transform:
        # Independent augmentations...
        for indx, img in enumerate(list_images):
            augmented = img_transform(image=img)
            list_images[indx] = augmented['image']

    if transform:
        # All augmentations applied in same proportion and values
        imgs_ids = ["image"] + ["image{}".format(idx + 2) for idx in range(len(list_images) - 1)]
        aug_args = dict(zip(imgs_ids, list_images))

        pair_ids_imgs = ["image{}".format(idx + 2) for idx in range(len(list_images) - 1)]
        base_id_imgs = ["image"] * len(pair_ids_imgs)
        list_additional_targets = dict(zip(pair_ids_imgs, base_id_imgs))

        volumetric_aug = albumentations.Compose(transform, additional_targets=list_additional_targets)
        augmented = volumetric_aug(**aug_args)

        list_images = np.stack([augmented[img] for img in imgs_ids])

    return list_images


def add_volume_depth_channels(list_images):
    b, d, h, w = list_images.shape
    new_list_images = torch.empty((b, 3, h, w))
    for indx, image in enumerate(list_images):
        new_list_images[indx, ...] = add_depth_channels(image)
    return new_list_images
