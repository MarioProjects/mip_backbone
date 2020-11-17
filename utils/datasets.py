from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from skimage import io
import albumentations
import copy

import utils.dataload as d


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

    def __init__(self, mode, transform, img_transform, add_depth=True, normalization="normalize"):
        """
        :param mode: (string) Dataset mode in ["train", "validation"]
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        """

        if mode not in ["train", "validation", "test"]:
            assert False, "Unknown mode '{}'".format(mode)

        self.base_dir = "data/LVSC"
        self.img_channels = 3
        self.class_to_cat = {1: "LV"}
        self.include_background = False
        self.num_classes = 1  # LV
        data, directory = [], ""

        if mode in ["train", "validation"]:
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

        if self.mode in ["validation", "test"]:  # 'image', 'original_img', 'original_mask', 'img_id'
            return {"image": image, "original_img": original_image, "original_mask": original_mask, "img_id": img_id}

        return {"image": image, "label": mask, "original_mask": original_mask}


class ACDC172Dataset(Dataset):
    """
    2D Dataset for ACDC Challenge.
    https://acdc.creatis.insa-lyon.fr/
    """

    def __init__(self, mode, transform, img_transform, add_depth=True, normalization="normalize"):
        """
        :param mode: (string) Dataset mode in ["train", "validation"]
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'global_standardize'
        """

        if mode not in ["train", "validation"]:
            assert False, "Unknown mode '{}'".format(mode)

        self.base_dir = "data/AC17"
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
        else:
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

        if self.mode == "validation":  # 'image', 'original_img', 'original_mask', 'img_id'
            return {"image": image, "original_img": original_image, "original_mask": original_mask, "img_id": img_id}

        return {"image": image, "label": mask, "original_mask": original_mask}


def dataset_selector(train_aug, train_aug_img, val_aug, args, is_test=False):
    if args.dataset == "DRIVE":
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
            add_depth=args.add_depth, normalization=args.normalization
        )

        val_dataset = LVSC2Dataset(
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
