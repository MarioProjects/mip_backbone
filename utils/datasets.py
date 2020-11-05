from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from skimage import io
import albumentations
import copy

import utils.dataload as d


def load_tif(tif_path):
    return np.array(Image.open(tif_path))


class DRIVEDataset(Dataset):
    """
    Dataset for Digital Retinal Images for Vessel Extraction (DRIVE) Challenge.
    https://drive.grand-challenge.org/
    """

    def __init__(self, mode, transform, img_transform, normalization="normalize"):
        """
        :param mode: (string) Dataset mode in ["train", "validation"]
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        """

        if mode not in ["train", "validation"]:
            assert False, "Unknown mode '{}'".format(mode)

        self.base_dir = "data/DRIVE"
        self.img_channels = 3
        self.class_to_cat = {0: "Background", 1: "Vessel"}
        self.include_background = True
        self.num_classes = 2  # background - vessel

        data = []
        directory = os.path.join(self.base_dir, "training", "images")
        for entry in os.scandir(directory):
            if entry.path.endswith(".tif"):
                data.append(entry.path)

        np.random.seed(1)
        np.random.shuffle(data)
        if mode == "train":
            data = data[:int(len(data) * .75)]
        else:
            data = data[int(len(data) * .75):]

        self.data = data
        self.mode = mode
        self.normalization = normalization

        self.transform = albumentations.Compose(transform)
        self.img_transform = albumentations.Compose(img_transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = self.data[idx]
        image = load_tif(img_path)

        img_id = img_path.split("/")[-1][0:2]
        mask_path = os.path.join(
            self.base_dir, "training", "1st_manual", img_id + "_manual1.gif"
        )
        mask = load_tif(mask_path)
        mask = np.where(mask > 0, 1, 0).astype(mask.dtype)  # Vessel values are 255, reconvert to 1

        original_image = copy.deepcopy(image)
        original_mask = copy.deepcopy(mask)

        image, mask = d.apply_augmentations(image, self.transform, self.img_transform, mask)
        image = d.apply_normalization(image, self.normalization).transpose(2, 0, 1)
        image = torch.from_numpy(image)

        mask = torch.from_numpy(mask).long()

        if self.mode == "validation":  # 'image', 'original_img', 'original_mask', 'img_id'
            return {"image": image, "original_img": original_image, "original_mask": original_mask, "img_id": img_id}

        return {"image": image, "label": mask, "original_mask": original_mask}


class SIMEPUSegmentationDataset(Dataset):
    """
    Dataset for Digital Retinal Images for Vessel Extraction (DRIVE) Challenge.
    https://drive.grand-challenge.org/
    """

    def __init__(self, mode, transform, img_transform, selected_class="", normalization="normalize"):
        """
        :param mode: (string) Dataset mode in ["train", "validation"]
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        """

        if selected_class == "":
            assert False, "Please specify a class to perform segmentation!"

        if mode not in ["train", "validation"]:
            assert False, "Unknown mode '{}'".format(mode)

        self.base_dir = "data/SIMEPU_Segmentation"
        self.img_channels = 3
        self.class_to_cat = {0: "Damage"}
        self.include_background = False
        self.num_classes = 1  # background - damage

        data = []
        directory = os.path.join(self.base_dir, "masks", selected_class)
        for entry in os.scandir(directory):
            if entry.path.endswith(".jpg"):
                data.append(entry.path)

        np.random.seed(1)
        np.random.shuffle(data)
        if mode == "train":
            data = data[:int(len(data) * .75)]
        else:
            data = data[int(len(data) * .75):]

        self.data = data
        self.mode = mode
        self.normalization = normalization

        self.transform = albumentations.Compose(transform)
        self.img_transform = albumentations.Compose(img_transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = self.data[idx].replace("masks", "images")
        image = io.imread(img_path)

        img_id = img_path.split("/")[-1][0:2]
        mask_path = self.data[idx]
        mask = np.where(io.imread(mask_path)[..., 0] > 0.5, 1, 0).astype(np.int32)

        original_image = copy.deepcopy(image)
        original_mask = copy.deepcopy(mask)

        image, mask = d.apply_augmentations(image, self.transform, self.img_transform, mask)
        image = d.apply_normalization(image, self.normalization).transpose(2, 0, 1)
        image = torch.from_numpy(image)

        mask = torch.from_numpy(mask).long()

        if self.mode == "validation":  # 'image', 'original_img', 'original_mask', 'img_id'
            return {"image": image, "original_img": original_image, "original_mask": original_mask, "img_id": img_id}

        return {"image": image, "label": mask, "original_mask": original_mask}


def dataset_selector(train_aug, train_aug_img, val_aug, args):
    if args.dataset == "DRIVE":
        train_dataset = DRIVEDataset(
            mode="train", transform=train_aug, img_transform=train_aug_img, add_depth=args.add_depth
        )

        val_dataset = DRIVEDataset(
            mode="validation", transform=val_aug, img_transform=[], normalization=args.normalization
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

    elif args.dataset == "SIMEPUSegmentation":
        train_dataset = SIMEPUSegmentationDataset(
            mode="train", transform=train_aug, img_transform=train_aug_img,
            selected_class=args.selected_class, normalization=args.normalization
        )

        val_dataset = SIMEPUSegmentationDataset(
            mode="validation", transform=val_aug, img_transform=[],
            selected_class=args.selected_class, normalization=args.normalization
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

    else:
        assert False, f"Unknown dataset '{args.dataset}'"

    print(f"Train dataset len:  {len(train_dataset)}")
    print(f"Validation dataset len:  {len(val_dataset)}")
    return train_loader, val_loader
