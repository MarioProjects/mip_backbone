""" --- DATA AUGMENTATION METHODS --- """

import albumentations


def data_augmentation_selector(da_policy, img_size, crop_size, mask_reshape_method):
    if da_policy == "none" or da_policy is None:
        return da_policy_none(img_size, mask_reshape_method)

    elif da_policy == "random_crops":
        return da_policy_randomcrop(img_size, crop_size, mask_reshape_method)

    elif da_policy == "rotations":
        return da_policy_rotate(img_size, mask_reshape_method)

    elif da_policy == "vflips":
        return da_policy_vflip(img_size, mask_reshape_method)

    elif da_policy == "hflips":
        return da_policy_hflip(img_size, mask_reshape_method)

    elif da_policy == "elastic_transform":
        return da_policy_elastictransform(img_size, mask_reshape_method)

    elif da_policy == "grid_distortion":
        return da_policy_griddistortion(img_size, mask_reshape_method)

    elif da_policy == "shift":
        return da_policy_shift(img_size, mask_reshape_method)

    elif da_policy == "scale":
        return da_policy_scale(img_size, mask_reshape_method)

    elif da_policy == "optical_distortion":
        return da_policy_opticaldistortion(img_size, mask_reshape_method)

    elif da_policy == "coarse_dropout" or da_policy == "cutout":
        return da_policy_coarsedropout(img_size, mask_reshape_method)

    elif da_policy == "downscale":
        return da_policy_downscale(img_size, mask_reshape_method)

    elif da_policy == "drive":
        return drive_da(img_size, mask_reshape_method)

    elif da_policy == "simepu_segmentation":
        return simepu_segmentation_da(img_size, mask_reshape_method)

    elif da_policy == "lvsc2d":
        return lvsc2d_segmentation_da(img_size, mask_reshape_method)

    elif da_policy == "acdc172d":
        return lvsc2d_segmentation_da(img_size, mask_reshape_method)

    assert False, "Unknown Data Augmentation Policy: {}".format(da_policy)


# ----------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------- #

def common_test_augmentation(img_size, mask_reshape_method):
    if mask_reshape_method == "padd":
        print("Padding masks!")
        return [
            albumentations.PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True),
            albumentations.CenterCrop(height=img_size, width=img_size, always_apply=True),
            albumentations.Resize(img_size, img_size, always_apply=True)
        ]
    elif mask_reshape_method == "resize":
        print("Resizing masks!")
        return [
            albumentations.Resize(img_size, img_size, always_apply=True)
        ]
    else:
        assert False, f"Unknown mask resize method '{mask_reshape_method}'"


def da_policy_none(img_size, mask_reshape_method):
    print("Using None Data Augmentation")
    train_aug = [
    ]

    train_aug = common_test_augmentation(img_size, mask_reshape_method) + train_aug

    train_aug_img = []

    val_aug = common_test_augmentation(img_size, mask_reshape_method)

    return train_aug, train_aug_img, val_aug


def da_policy_randomcrop(img_size, crop_size, mask_reshape_method):  # DA_Rotate
    print("Using Data Augmentation Random Crops")
    train_aug = [
        albumentations.RandomCrop(height=crop_size, width=crop_size)
    ]

    train_aug = common_test_augmentation(img_size, mask_reshape_method) + train_aug

    train_aug_img = []

    val_aug = common_test_augmentation(crop_size)

    return train_aug, train_aug_img, val_aug


def da_policy_rotate(img_size, mask_reshape_method):  # DA_Rotate
    print("Using Data Augmentation Rotations")
    train_aug = [
        albumentations.Rotate(p=0.55, limit=45, interpolation=1, border_mode=0),
    ]

    train_aug = common_test_augmentation(img_size, mask_reshape_method) + train_aug

    train_aug_img = []

    val_aug = common_test_augmentation(img_size, mask_reshape_method)

    return train_aug, train_aug_img, val_aug


def da_policy_vflip(img_size, mask_reshape_method):
    print("Using Data Augmentation Vertical Flips")
    train_aug = [
        albumentations.VerticalFlip(p=0.5),
    ]

    train_aug = common_test_augmentation(img_size, mask_reshape_method) + train_aug

    train_aug_img = []

    val_aug = common_test_augmentation(img_size, mask_reshape_method)

    return train_aug, train_aug_img, val_aug


def da_policy_hflip(img_size, mask_reshape_method):
    print("Using Data Augmentation Horizontal Flips")
    train_aug = [
        albumentations.HorizontalFlip(p=0.5),
    ]

    train_aug = common_test_augmentation(img_size, mask_reshape_method) + train_aug

    train_aug_img = []

    val_aug = common_test_augmentation(img_size, mask_reshape_method)

    return train_aug, train_aug_img, val_aug


def da_policy_elastictransform(img_size, mask_reshape_method):
    print("Using Data Augmentation ElasticTransform")
    train_aug = [
        albumentations.ElasticTransform(p=0.7, alpha=333, sigma=333 * 0.05, alpha_affine=333 * 0.1),
    ]

    train_aug = common_test_augmentation(img_size, mask_reshape_method) + train_aug

    train_aug_img = []

    val_aug = common_test_augmentation(img_size, mask_reshape_method)

    return train_aug, train_aug_img, val_aug


def da_policy_griddistortion(img_size, mask_reshape_method):
    print("Using Data Augmentation GridDistortion")
    train_aug = [
        albumentations.GridDistortion(p=0.55, distort_limit=0.5),
    ]

    train_aug = common_test_augmentation(img_size, mask_reshape_method) + train_aug

    train_aug_img = []

    val_aug = common_test_augmentation(img_size, mask_reshape_method)

    return train_aug, train_aug_img, val_aug


def da_policy_shift(img_size, mask_reshape_method):
    print("Using Data Augmentation Shift")
    train_aug = [
        albumentations.ShiftScaleRotate(p=0.65, shift_limit=0.2, scale_limit=0.0, rotate_limit=0)
    ]

    train_aug = common_test_augmentation(img_size, mask_reshape_method) + train_aug

    train_aug_img = []

    val_aug = common_test_augmentation(img_size, mask_reshape_method)

    return train_aug, train_aug_img, val_aug


def da_policy_scale(img_size, mask_reshape_method):
    print("Using Data Augmentation Scale")
    train_aug = [
        albumentations.ShiftScaleRotate(p=0.65, shift_limit=0.0, scale_limit=0.2, rotate_limit=0)
    ]

    train_aug = common_test_augmentation(img_size, mask_reshape_method) + train_aug

    train_aug_img = []

    val_aug = common_test_augmentation(img_size, mask_reshape_method)

    return train_aug, train_aug_img, val_aug


def da_policy_opticaldistortion(img_size, mask_reshape_method):
    print("Using Data Augmentation OpticalDistortion")
    train_aug = [
        albumentations.OpticalDistortion(p=0.6, distort_limit=0.7, shift_limit=0.2)
    ]

    train_aug = common_test_augmentation(img_size, mask_reshape_method) + train_aug

    train_aug_img = []

    val_aug = common_test_augmentation(img_size, mask_reshape_method)

    return train_aug, train_aug_img, val_aug


def da_policy_coarsedropout(img_size, mask_reshape_method):
    print("Using Data Augmentation CoarseDropout")
    train_aug = [
        albumentations.CoarseDropout(p=0.6, max_holes=3, max_height=25, max_width=25)
    ]

    train_aug = common_test_augmentation(img_size, mask_reshape_method) + train_aug

    train_aug_img = []

    val_aug = common_test_augmentation(img_size, mask_reshape_method)

    return train_aug, train_aug_img, val_aug


def da_policy_downscale(img_size, mask_reshape_method):
    print("Using Data Augmentation Downscale")
    train_aug = [
    ]

    train_aug = common_test_augmentation(img_size, mask_reshape_method) + train_aug

    train_aug_img = [albumentations.Downscale(p=0.7, scale_min=0.4, scale_max=0.8, interpolation=0)]

    val_aug = common_test_augmentation(img_size, mask_reshape_method)

    return train_aug, train_aug_img, val_aug


###########################################
#  --- DATA AUGMENTATION COMBINATIONS --- #
###########################################

def drive_da(img_size, mask_reshape_method):
    print("Using DRIVE Data Augmentation Combinations")
    train_aug = [
        albumentations.ElasticTransform(p=0.72, alpha=177, sigma=177 * 0.05, alpha_affine=176 * 0.03),
        albumentations.GridDistortion(p=0.675, distort_limit=0.3),
        albumentations.OpticalDistortion(p=0.2, distort_limit=0.2, shift_limit=0.2),

        albumentations.ShiftScaleRotate(p=0.56, shift_limit=0.2, scale_limit=0.0, rotate_limit=0),  # shift
        albumentations.ShiftScaleRotate(p=0.25, shift_limit=0.0, scale_limit=0.2, rotate_limit=0),  # scale

        albumentations.VerticalFlip(p=0.325),
        albumentations.HorizontalFlip(p=0.3),
        albumentations.Rotate(p=0.625, limit=45, interpolation=1, border_mode=0),
    ]

    train_aug = common_test_augmentation(img_size, mask_reshape_method) + train_aug

    train_aug_img = []

    val_aug = common_test_augmentation(img_size, mask_reshape_method)

    return train_aug, train_aug_img, val_aug


def simepu_segmentation_da(img_size, mask_reshape_method):
    print("Using SIMEPU Segmentation Data Augmentation Combinations")
    train_aug = [
        albumentations.ElasticTransform(p=0.72, alpha=177, sigma=177 * 0.05, alpha_affine=176 * 0.03),
        albumentations.GridDistortion(p=0.675, distort_limit=0.3),
        albumentations.OpticalDistortion(p=0.2, distort_limit=0.2, shift_limit=0.2),

        albumentations.ShiftScaleRotate(p=0.56, shift_limit=0.2, scale_limit=0.0, rotate_limit=0),  # shift
        albumentations.ShiftScaleRotate(p=0.25, shift_limit=0.0, scale_limit=0.2, rotate_limit=0),  # scale

        albumentations.VerticalFlip(p=0.325),
        albumentations.HorizontalFlip(p=0.3),
    ]

    train_aug = common_test_augmentation(img_size, mask_reshape_method) + train_aug

    train_aug_img = []

    val_aug = common_test_augmentation(img_size, mask_reshape_method)

    return train_aug, train_aug_img, val_aug


def lvsc2d_segmentation_da(img_size, mask_reshape_method):
    print("Using LVSC 2D Segmentation Data Augmentation Combinations")
    train_aug = [
        albumentations.ElasticTransform(p=0.72, alpha=177, sigma=177 * 0.05, alpha_affine=176 * 0.03),
        albumentations.GridDistortion(p=0.675, distort_limit=0.3),
        albumentations.OpticalDistortion(p=0.2, distort_limit=0.2, shift_limit=0.2),

        albumentations.ShiftScaleRotate(p=0.56, shift_limit=0.2, scale_limit=0.0, rotate_limit=0),  # shift
        albumentations.ShiftScaleRotate(p=0.25, shift_limit=0.0, scale_limit=0.2, rotate_limit=0),  # scale

        albumentations.VerticalFlip(p=0.325),
        albumentations.HorizontalFlip(p=0.3),
        albumentations.Rotate(p=0.625, limit=45, interpolation=1, border_mode=0),
    ]

    train_aug = common_test_augmentation(img_size, mask_reshape_method) + train_aug

    train_aug_img = []

    val_aug = common_test_augmentation(img_size, mask_reshape_method)

    return train_aug, train_aug_img, val_aug
