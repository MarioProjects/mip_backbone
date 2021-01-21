from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from skimage import io
import albumentations
import copy
import pandas as pd

import utils.dataload as d


class MMs2DDataset(Dataset):
    """
    Dataset for Digital Retinal Images for Vessel Extraction (DRIVE) Challenge.
    https://drive.grand-challenge.org/
    """

    def __init__(self, partition, transform, img_transform, normalization="normalize", add_depth=True,
                 is_labeled=True, centre=None, vendor=None, end_volumes=True, data_relative_path=""):
        """
        :param partition: (string) Dataset partition in ["Training", "Validation", "Test"]
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        :param add_depth: (bool) Whether transform or not 1d slices to 3 channels images
        :param is_labeled: (bool) Dataset partition in ["Training", "Validation", "Test"]
        :param centre: (int) Select by centre label. Available [1, 2, 3, 4, 5]
        :param vendor: (string) Select by vendor label. Available ["A", "B", "C", "D"]
        :param end_volumes: (bool) Whether only include 'ED' and 'ES' phases ((to) segmented) or all
        :param data_relative_path: (string) Prepend extension to MMs data base dir
        """

        if partition not in ["Training", "Validation", "Testing"]:
            assert False, "Unknown mode '{}'".format(partition)

        self.base_dir = os.path.join(data_relative_path, "data/MMs")
        self.partition = partition
        self.img_channels = 3
        self.class_to_cat = {1: "LV", 2: "MYO", 3: "RV", 4: "Mean"}
        self.include_background = False
        self.num_classes = 4  # background - LV - MYO - RV

        data = pd.read_csv(os.path.join(self.base_dir, "slices_info.csv"))
        data = data.loc[(data["Partition"] == partition) & (data["Labeled"] == is_labeled)]
        if vendor is not None:
            data = data.loc[data['Vendor'].isin(vendor)]
        if centre is not None:
            data = data.loc[data['Centre'].isin(centre)]

        if end_volumes:  # Get only volumes in 'ED' and 'ES' phases (segmented)
            data = data.loc[(data["ED"] == data["Phase"]) | (data["ES"] == data["Phase"])]

        data = data.reset_index(drop=True)
        self.data = data

        self.add_depth = add_depth
        self.normalization = normalization
        self.transform = albumentations.Compose(transform)
        self.img_transform = albumentations.Compose(img_transform)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def custom_collate(batch):
        """

        Args:
            batch: list of dataset items (from __getitem__). In this case batch is a list of dicts with
                   key image, and depending of validation or train different keys

        Returns:

        """
        # We have to modify "original_mask" as has different shapes
        batch_keys = list(batch[0].keys())
        res = {bkey: [] for bkey in batch_keys}
        for belement in batch:
            for bkey in batch_keys:
                res[bkey].append(belement[bkey])

        for bkey in batch_keys:
            if bkey == "original_mask" or bkey == "original_img" or bkey == "img_id":
                continue  # We wont stack over original_mask...
            res[bkey] = torch.stack(res[bkey]) if None not in res[bkey] else None

        return res

    def __getitem__(self, idx):
        df_entry = self.data.loc[idx]
        external_code = df_entry["External code"]
        c_slice = df_entry["Slice"]
        c_phase = df_entry["Phase"]
        c_vendor = df_entry["Vendor"]
        c_centre = df_entry["Centre"]
        img_id = f"{external_code}_slice{c_slice}_phase{c_phase}_vendor{c_vendor}_centre{c_centre}"

        labeled_info = ""
        if self.partition == "Training":
            labeled_info = "Labeled" if df_entry["Labeled"] else "Unlabeled"

        img_path = os.path.join(
            self.base_dir, self.partition, labeled_info, external_code,
            f"{external_code}_sa_slice{c_slice}_phase{c_phase}.npy"
        )
        image = np.load(img_path)

        mask = None
        if not (self.partition == "Training" and not df_entry["Labeled"]):
            mask_path = os.path.join(
                self.base_dir, self.partition, labeled_info, external_code,
                f"{external_code}_sa_gt_slice{c_slice}_phase{c_phase}.npy"
            )
            mask = np.load(mask_path)

        original_image = copy.deepcopy(image)
        original_mask = copy.deepcopy(mask)

        image, mask = d.apply_augmentations(image, self.transform, self.img_transform, mask)
        image = d.apply_normalization(image, self.normalization)
        image = torch.from_numpy(np.expand_dims(image, axis=0))

        if self.add_depth:
            image = d.add_depth_channels(image)
        mask = torch.from_numpy(np.expand_dims(mask, 0)).float() if mask is not None else None

        return {
            "img_id": img_id, "image": image, "label": mask,
            "original_img": original_image, "original_mask": original_mask
        }


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
        self.class_to_cat = {1: "Vessel"}
        self.include_background = False
        self.num_classes = 1  # background - vessel

        data = []
        directory = os.path.join(self.base_dir, "training", "images")
        for entry in os.scandir(directory):
            if entry.path.endswith(".tif"):
                data.append(entry.path)

        np.random.seed(1)
        np.random.shuffle(data)
        if mode == "train":
            data = data[:int(len(data) * 85)]
        else:
            data = data[int(len(data) * .85):]

        self.data = data
        self.mode = mode
        self.normalization = normalization

        self.transform = albumentations.Compose(transform)
        self.img_transform = albumentations.Compose(img_transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = self.data[idx]
        image = d.load_tif(img_path)

        img_id = img_path.split("/")[-1][0:2]
        mask_path = os.path.join(
            self.base_dir, "training", "1st_manual", img_id + "_manual1.gif"
        )
        mask = d.load_tif(mask_path)
        mask = np.where(mask > 0, 1, 0).astype(mask.dtype)  # Vessel values are 255, reconvert to 1

        original_image = copy.deepcopy(image)
        original_mask = copy.deepcopy(mask)

        image, mask = d.apply_augmentations(image, self.transform, self.img_transform, mask)
        image = d.apply_normalization(image, self.normalization).transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        mask = torch.from_numpy(np.expand_dims(mask, 0)).float()

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
        self.class_to_cat = {1: "Daño"}
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
            data = data[:int(len(data) * .85)]
        else:
            data = data[int(len(data) * .85):]

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

        img_id = os.path.splitext(img_path)[0].split("/")[-1]
        mask_path = self.data[idx]
        mask = np.where(io.imread(mask_path)[..., 0] > 0.5, 1, 0).astype(np.int32)

        original_image = copy.deepcopy(image)
        original_mask = copy.deepcopy(mask)

        image, mask = d.apply_augmentations(image, self.transform, self.img_transform, mask)
        image = d.apply_normalization(image, self.normalization)

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(np.expand_dims(mask, 0)).float()

        if self.mode == "validation":  # 'image', 'original_img', 'original_mask', 'img_id'
            return {"image": image, "original_img": original_image, "original_mask": original_mask, "img_id": img_id}

        return {"image": image, "label": mask, "original_mask": original_mask}


class LVSC2Dataset(Dataset):
    """
    2D Dataset for LVSC Challenge.
    """

    def __init__(self, mode, transform, img_transform, add_depth=True, normalization="normalize", relative_path="",
                 train_patients=100):
        """
        :param mode: (string) Dataset mode in ["train", "validation"]
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        """

        if mode not in ["train", "full_train", "validation", "test"]:
            assert False, "Unknown mode '{}'".format(mode)

        self.base_dir = os.path.join(relative_path, "data/LVSC")
        self.img_channels = 3
        self.class_to_cat = {1: "LV"}
        self.include_background = False
        self.num_classes = 1  # LV
        data, directory = [], ""

        if mode in ["train", "full_train", "validation"]:
            directory = os.path.join(self.base_dir, "Training")
        elif mode in ["test"]:
            directory = os.path.join(self.base_dir, "Validation")

        for subdir, dirs, files in os.walk(directory):
            for file in files:
                entry = os.path.join(subdir, file)
                if "_SA" in entry and entry.endswith(".dcm"):
                    # Check that we have corresponding mask, not all masks are available
                    if mode == "test" and not os.path.isfile(self.test_dcm2png_path(entry)):
                        continue
                    data.append(entry)

        np.random.seed(1)
        patient_list = np.sort(np.unique([elem.split("/")[-2] for elem in data]))
        train_indx = np.random.choice(range(patient_list.shape[0]), size=(train_patients,), replace=False)
        ind = np.zeros(patient_list.shape[0], dtype=bool)
        ind[train_indx] = True
        val_indx = ~ind

        np.random.shuffle(data)
        if mode == "train":
            data = [elem for elem in data if elem.split("/")[-2] in patient_list[train_indx]]
        elif mode == "validation":
            if train_patients > 85:
                # If there are not too much patients take randomly
                np.random.seed(1)
                np.random.shuffle(data)
                data = data[int(len(data) * .85):]
            else:
                # Only get first 15 patients for validation, not ALL
                data = [elem for elem in data if elem.split("/")[-2] in patient_list[val_indx][:15]]

        self.data = data
        self.mode = mode
        self.normalization = normalization

        self.transform = albumentations.Compose(transform)
        self.img_transform = albumentations.Compose(img_transform)
        self.add_depth = add_depth

    def __len__(self):
        return len(self.data)

    @staticmethod
    def custom_collate(batch):
        """

        Args:
            batch: list of dataset items (from __getitem__). In this case batch is a list of dicts with
                   key image, and depending of validation or train different keys

        Returns:

        """
        # We have to modify "original_mask" as has different shapes
        batch_keys = list(batch[0].keys())
        res = {bkey: [] for bkey in batch_keys}
        for belement in batch:
            for bkey in batch_keys:
                res[bkey].append(belement[bkey])

        for bkey in batch_keys:
            if bkey == "original_mask" or bkey == "original_img" or bkey == "img_id":
                continue  # We wont stack over original_mask...
            res[bkey] = torch.stack(res[bkey])

        return res

    @staticmethod
    def test_dcm2png_path(dcm_path):
        return "/".join([
            str(x) for x in
            dcm_path.replace(".dcm", ".png").replace("Validation", "consensus/images").split("/")
            if "CAP" not in x
        ])

    def __getitem__(self, idx):

        img_path = self.data[idx]
        image = d.read_dicom(img_path)

        img_id = os.path.splitext(img_path)[0].split("/")[-1]
        if self.mode == "test":
            mask_path = self.test_dcm2png_path(self.data[idx])
            mask = np.where(io.imread(mask_path) > 0.5, 1, 0).astype(np.int32)
        else:
            mask_path = self.data[idx].replace(".dcm", ".png")
            mask = np.where(io.imread(mask_path)[..., 0] > 0.5, 1, 0).astype(np.int32)

        original_image = copy.deepcopy(image)
        original_mask = copy.deepcopy(mask)

        image, mask = d.apply_augmentations(image, self.transform, self.img_transform, mask)
        image = d.apply_normalization(image, self.normalization)
        image = torch.from_numpy(np.expand_dims(image, axis=0))

        if self.add_depth:
            image = d.add_depth_channels(image)
        mask = torch.from_numpy(np.expand_dims(mask, 0)).float()

        return {
            "image": image, "original_img": original_image, "label": mask,
            "original_mask": original_mask, "img_id": img_id
        }


class ACDC172Dataset(Dataset):
    """
    2D Dataset for ACDC Challenge.
    https://acdc.creatis.insa-lyon.fr/
    """

    def __init__(self, mode, transform, img_transform, add_depth=True, normalization="normalize", relative_path=""):
        """
        :param mode: (string) Dataset mode in ["train", "validation"]
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        """

        if mode not in ["train", "full_train", "validation"]:
            assert False, "Unknown mode '{}'".format(mode)

        self.base_dir = os.path.join(relative_path, "data/AC17")
        self.img_channels = 3
        self.class_to_cat = {1: "RV", 2: "MYO", 3: "LV", 4: "Mean"}
        self.num_classes = 4
        self.include_background = False

        data = []
        for subdir, dirs, files in os.walk(self.base_dir):
            for file in files:
                entry = os.path.join(subdir, file)
                if "_gt" in entry and not ".nii" in entry and not ".nii.gz" in entry:
                    data.append(entry)

        if len(data) == 0:
            assert False, 'You have to transform volumes to 2D slices: ' \
                          'python tools/nifti2slices.py --data_path "data/AC17"'

        np.random.seed(1)
        np.random.shuffle(data)
        if mode == "train":
            data = data[:int(len(data) * .85)]
        elif mode == "validation":
            data = data[int(len(data) * .85):]

        self.data = data
        self.mode = mode
        self.normalization = normalization

        self.transform = albumentations.Compose(transform)
        self.img_transform = albumentations.Compose(img_transform)
        self.add_depth = add_depth

    def __len__(self):
        return len(self.data)

    @staticmethod
    def custom_collate(batch):
        """

        Args:
            batch: list of dataset items (from __getitem__). In this case batch is a list of dicts with
                   key image, and depending of validation or train different keys

        Returns:

        """
        # We have to modify "original_mask" as has different shapes
        batch_keys = list(batch[0].keys())
        res = {bkey: [] for bkey in batch_keys}
        for belement in batch:
            for bkey in batch_keys:
                res[bkey].append(belement[bkey])

        for bkey in batch_keys:
            if bkey == "original_mask" or bkey == "original_img" or bkey == "img_id":
                continue  # We wont stack over original_mask...
            res[bkey] = torch.stack(res[bkey])

        return res

    def __getitem__(self, idx):

        img_path = self.data[idx].replace("_gt", "")
        image = np.load(img_path)

        mask_path = self.data[idx]
        mask = np.load(mask_path)

        img_id = os.path.splitext(img_path)[0].split("/")[-1]

        original_image = copy.deepcopy(image)
        original_mask = copy.deepcopy(mask)

        image, mask = d.apply_augmentations(image, self.transform, self.img_transform, mask)
        image = d.apply_normalization(image, self.normalization)
        image = torch.from_numpy(np.expand_dims(image, axis=0))

        if self.add_depth:
            image = d.add_depth_channels(image)
        mask = torch.from_numpy(np.expand_dims(mask, 0)).float()

        return {
            "image": image, "original_img": original_image,
            "original_mask": original_mask, "label": mask, "img_id": img_id
        }


class ACDC173Dataset(Dataset):
    """
    3D Dataset for ACDC Challenge.
    https://acdc.creatis.insa-lyon.fr/
    """

    def __init__(self, mode, transform, img_transform, add_depth=True, normalization="normalize", relative_path=""):
        """
        :param mode: (string) Dataset mode in ["train", "validation"]
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        """

        if mode not in ["train", "full_train", "validation"]:
            assert False, "Unknown mode '{}'".format(mode)

        self.base_dir = os.path.join(relative_path, "data/AC17")
        self.img_channels = 3
        self.class_to_cat = {1: "RV", 2: "MYO", 3: "LV", 4: "Mean"}
        self.num_classes = 4
        self.include_background = False

        data = []
        for subdir, dirs, files in os.walk(self.base_dir):
            for file in files:
                entry = os.path.join(subdir, file)
                if "_gt" in entry and (".nii" in entry or ".nii.gz" in entry):
                    data.append(entry)

        if len(data) == 0:
            assert False, 'You have to transform volumes to 2D slices: ' \
                          'python tools/nifti2slices.py --data_path "data/AC17"'

        np.random.seed(1)
        np.random.shuffle(data)
        if mode == "train":
            data = data[:int(len(data) * .85)]
        elif mode == "validation":
            data = data[int(len(data) * .85):]

        self.data = data
        self.mode = mode
        self.normalization = normalization

        self.transform = albumentations.Compose(transform)
        self.img_transform = albumentations.Compose(img_transform)
        self.add_depth = add_depth

    def __len__(self):
        return len(self.data)

    @staticmethod
    def custom_collate(batch):
        """

        Args:
            batch: list of dataset items (from __getitem__). In this case batch is a list of dicts with
                   key image, and depending of validation or train different keys

        Returns:

        """
        # We have to modify "original_mask" as has different shapes
        not_stack = ["original_mask", "original_img", "img_id", "phase",
                     "mask_header", "mask_affine", "vol_header", "vol_affine"]
        batch_keys = list(batch[0].keys())
        res = {bkey: [] for bkey in batch_keys}
        for belement in batch:
            for bkey in batch_keys:
                res[bkey].append(belement[bkey])

        for bkey in batch_keys:
            if bkey in not_stack:
                continue  # We wont stack over original_mask...
            res[bkey] = torch.stack(res[bkey])

        return res

    def __getPhase(self, info_path, volume_path):
        # Using readlines()
        file1 = open(info_path, 'r')
        Lines = file1.readlines()

        ed, es = 0, 0
        for line in Lines:
            line_info = line.strip()
            ed = int(line_info.split(":")[-1]) if "ED" in line_info else ed
            es = int(line_info.split(":")[-1]) if "ES" in line_info else es

        vol_phase = int(volume_path[volume_path.find("frame")+len("frame"):-7])  # -7 due to .nii.gz
        if ed == vol_phase:
            return "ED"
        elif es == vol_phase:
            return "ES"
        else:
            assert False, f"Unknown phase for {volume_path}"

    def __getitem__(self, idx):

        volume_path = self.data[idx].replace("_gt", "")
        volume, affine, header = d.load_nii(volume_path)  # This volume is from a phase [height, width, slices]
        volume = volume.transpose(2, 0, 1)

        volume_info_path = os.path.join("/".join(self.data[idx].split("/")[:-1]), "Info.cfg")
        vol_phase = self.__getPhase(volume_info_path, volume_path)

        mask_path = self.data[idx]
        mask, mask_affine, mask_header = d.load_nii(mask_path)
        mask = mask.transpose(2, 0, 1)

        img_id = self.data[idx].split("/")[-1][:-10]

        original_volume = copy.deepcopy(volume)
        original_mask = copy.deepcopy(mask)

        volume, mask = d.apply_volume_2Daugmentations(volume, self.transform, self.img_transform, mask)
        volume = d.apply_normalization(volume, self.normalization)

        # We have to stack volume as batch
        volume = np.expand_dims(volume, axis=0) if not self.add_depth else volume
        volume = torch.from_numpy(volume)

        if self.add_depth:
            volume = d.add_volume_depth_channels(volume.unsqueeze(1))
        mask = torch.from_numpy(np.expand_dims(mask, 0)).float() if mask is not None else None

        return {
            "image": volume, "original_img": original_volume, "vol_affine": affine, "vol_header": header,
            "original_mask": original_mask, "label": mask, "mask_affine": mask_affine, "mask_header": mask_header,
            "img_id": img_id, "phase": vol_phase
        }


def find_values(string, label, label_type):
    """

    Args:
        string:
        label:
        label_type:

    Returns:

    Example:
        string = "mms_centre14_vendorA"
        label = "centre"
        label_type = int
        -> res = [1, 4]

        string = "mms_centre14_vendorA"
        label = "vendor"
        label_type = str
        -> res = ['A']

    """
    res = None
    if string.find(label) != -1:
        c_centre = string[string.find(label) + len(label):]
        centre_break = c_centre.find("_")
        if centre_break != -1:
            c_centre = c_centre[:centre_break]
        res = [label_type(i) for i in c_centre]
    return res


def dataset_selector(train_aug, train_aug_img, val_aug, args, is_test=False):
    if "mms2d" in args.dataset:
        if is_test:
            test_dataset = MMs2DDataset(
                partition="Testing", transform=val_aug, img_transform=train_aug_img, vendor=None, end_volumes=True,
                normalization=args.normalization, add_depth=args.add_depth, is_labeled=False, centre=None,
            )

            return DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                drop_last=False, collate_fn=test_dataset.custom_collate
            )

        only_end = False if "full" in args.dataset else True
        unlabeled = True if "unlabeled" in args.dataset else False
        c_centre, c_vendor = find_values(args.dataset, "centre", int), find_values(args.dataset, "vendor", str)

        train_dataset = MMs2DDataset(
            partition="Training", transform=train_aug, img_transform=train_aug_img, normalization=args.normalization,
            add_depth=args.add_depth, is_labeled=(not unlabeled), centre=c_centre, vendor=c_vendor, end_volumes=only_end
        )

        val_dataset = MMs2DDataset(
            partition="Validation", transform=val_aug, img_transform=[], normalization=args.normalization,
            add_depth=args.add_depth, is_labeled=False, centre=None, vendor=None, end_volumes=True
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, pin_memory=True,
            shuffle=True, collate_fn=train_dataset.custom_collate
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
            drop_last=False, collate_fn=val_dataset.custom_collate
        )

    elif args.dataset == "DRIVE":
        if is_test:
            assert False, "Not test partition available"
        train_dataset = DRIVEDataset(
            mode="train", transform=train_aug, img_transform=train_aug_img, normalization=args.normalization
        )

        val_dataset = DRIVEDataset(
            mode="validation", transform=val_aug, img_transform=[], normalization=args.normalization
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)

    elif args.dataset == "SIMEPUSegmentation":
        if is_test:
            assert False, "Not test partition available"
        train_dataset = SIMEPUSegmentationDataset(
            mode="train", transform=train_aug, img_transform=train_aug_img,
            selected_class=args.selected_class, normalization=args.normalization
        )

        val_dataset = SIMEPUSegmentationDataset(
            mode="validation", transform=val_aug, img_transform=[],
            selected_class=args.selected_class, normalization=args.normalization
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)

    elif args.dataset == "LVSC2D":
        if is_test:
            test_dataset = LVSC2Dataset(
                mode="test", transform=val_aug, img_transform=[],
                add_depth=args.add_depth, normalization=args.normalization
            )

            return DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                drop_last=False, collate_fn=test_dataset.custom_collate
            )

        train_dataset = LVSC2Dataset(
            mode="train", transform=train_aug, img_transform=train_aug_img,
            add_depth=args.add_depth, normalization=args.normalization, train_patients=args.lv_train_patients
        )

        val_dataset = LVSC2Dataset(
            mode="validation", transform=val_aug, img_transform=[],
            add_depth=args.add_depth, normalization=args.normalization, train_patients=args.lv_train_patients
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, pin_memory=True,
            shuffle=True, collate_fn=train_dataset.custom_collate
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
            drop_last=False, collate_fn=val_dataset.custom_collate
        )

    elif args.dataset == "ACDC172D":
        if is_test:
            assert False, "Not test partition available"
        train_dataset = ACDC172Dataset(
            mode="train", transform=train_aug, img_transform=train_aug_img,
            add_depth=args.add_depth, normalization=args.normalization
        )

        val_dataset = ACDC172Dataset(
            mode="validation", transform=val_aug, img_transform=[],
            add_depth=args.add_depth, normalization=args.normalization
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, pin_memory=True,
            shuffle=True, collate_fn=train_dataset.custom_collate
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
            drop_last=False, collate_fn=val_dataset.custom_collate
        )

    else:
        assert False, f"Unknown dataset '{args.dataset}'"

    print(f"Train dataset len:  {len(train_dataset)}")
    print(f"Validation dataset len:  {len(val_dataset)}")
    return train_loader, val_loader
