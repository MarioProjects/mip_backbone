#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../')

from tqdm.notebook import tqdm
from pathlib import Path
import warnings

from models import model_selector
from utils.data_augmentation import data_augmentation_selector

from utils.neural import *
from utils.datasets import *
from utils.metrics import *

warnings.filterwarnings('ignore')

preds_dir = "ACDC_OverMMsSiemens_Analisis"
os.makedirs(preds_dir, exist_ok=True)


def map_mask_classes(mask, classes_map):
    """

    Args:
        mask: (np.array) Mask Array to map (height, width)
        classes_map: (dict) Mapping between classes. E.g.  {0:0, 1:3, 2:2, 3:1 ,4:4}

    Returns: (np.array) Mapped mask array

    """
    res = np.zeros_like(mask).astype(mask.dtype)
    for value in np.unique(mask):
        if value not in classes_map:
            assert False, f"Please specify all class maps. {value} not in {classes_map}"
        res += np.where(mask == value, classes_map[value], 0).astype(mask.dtype)
    return res


def find_path(directory, filename):
    for path in Path(directory).rglob(filename):
        return path

_, _, val_aug = data_augmentation_selector("none", 224, 224, "padd")

batch_size = 1
add_depth = True
normalization = "standardize"

train_dataset = ALLMMsDataset3D(
    transform=val_aug, img_transform=[],
    add_depth=add_depth, normalization=normalization, data_relative_path="../../mnms_da",
    vendor=["A"]  # Solo queremos comparar con mismo fabricante -> Siemens == Vendor A
)

allmms_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
    drop_last=False, collate_fn=train_dataset.simple_collate
)

metrics = {
    'img_id': [], 'phase': [],
    'iou_RV': [], 'dice_RV': [], 'hd_RV': [], 'assd_RV': [],
    'iou_MYO': [], 'dice_MYO': [], 'hd_MYO': [], 'assd_MYO': [],
    'iou_LV': [], 'dice_LV': [], 'hd_LV': [], 'assd_LV': [],
}

# MMS -> class_to_cat = {1: "LV", 2: "MYO", 3: "RV"}
# Original ACDC -> class_to_cat = {1: "RV", 2: "MYO", 3: "LV"}
map_classes = {0: 0, 1: 3, 2: 2, 3: 1}
class_to_cat = {1: "RV", 2: "MYO", 3: "LV"}
mask_reshape_method = "padd"
include_background = False

LISTA_CHECKPOINTS = ["n0_5.pt", "n0_5_swa.pt", "n0_10.pt", "n0_10_swa.pt", "n0_25.pt", "n0_25_swa.pt",
                     "n0_50.pt", "n0_50_swa.pt", "n0_100.pt", "n0_100_swa.pt", "n1_5.pt", "n1_5_swa.pt",
                     "n1_10.pt", "n1_10_swa.pt", "n1_25.pt", "n1_25_swa.pt", "n1_50.pt", "n1_50_swa.pt",
                     "n1_100.pt", "n1_100_swa.pt"]
                     
LISTA_CHECKPOINTS = [ "n1_50_swa.pt", "n1_100.pt", "n1_100_swa.pt"]

for check in LISTA_CHECKPOINTS:
    model = None
    model = model_selector(
        "segmentation", "resnet34_unet_imagenet_encoder_scse_hypercols", num_classes=4, from_swa=True if "swa" in check else False,
        in_channels=3, devices=[0], checkpoint=f"../checks/ACDC/{check}"
    )

    model.eval()

    with torch.no_grad():
        for batch_indx, batch in enumerate(allmms_loader):
            img_id = batch["external_code"]
            partition = batch["partition"]
            init_shape = batch["initial_shape"]

            ed_vol = batch["ed_volume"].squeeze().cuda()
            es_vol = batch["es_volume"].squeeze().cuda()

            original_ed = batch["original_ed"]
            original_es = batch["original_es"]

            original_ed_mask = batch["original_ed_mask"]
            original_es_mask = batch["original_es_mask"]

            mask_affine = batch["affine"]
            mask_header = batch["header"]

            # ---- ED CASE ----

            prob_preds = model(ed_vol)
            pred_mask = convert_multiclass_mask(prob_preds).data.cpu().numpy()
            pred_mask = pred_mask.astype(np.uint8)
            pred_mask = map_mask_classes(pred_mask, map_classes)
            pred_mask = reshape_volume(pred_mask, init_shape[:2], mask_reshape_method)
            pred_mask = pred_mask.transpose(1, 2, 0)

            for current_class in range(len(map_classes)):

                if not include_background and current_class == 0:
                    continue

                y_true = np.where(original_ed_mask == current_class, 1, 0).astype(np.int32)
                y_pred = np.where(pred_mask == current_class, 1, 0).astype(np.int32)
                class_str = class_to_cat[current_class]

                jc_score = jaccard_coef(y_true, y_pred)
                dc_score = dice_coef(y_true, y_pred)
                hd_score = secure_hd(y_true, y_pred)
                assd_score = secure_assd(y_true, y_pred)

                metrics[f'iou_{class_str}'].append(jc_score)
                metrics[f'dice_{class_str}'].append(dc_score)
                metrics[f'hd_{class_str}'].append(hd_score)
                metrics[f'assd_{class_str}'].append(assd_score)

            metrics[f'img_id'].append(img_id)
            metrics[f'phase'].append("ED")

            # ---- ES CASE ----

            prob_preds = model(es_vol)
            pred_mask = convert_multiclass_mask(prob_preds).data.cpu().numpy()
            pred_mask = pred_mask.astype(np.uint8)
            pred_mask = map_mask_classes(pred_mask, map_classes)
            pred_mask = reshape_volume(pred_mask, init_shape[:2], mask_reshape_method)
            pred_mask = pred_mask.transpose(1, 2, 0)

            for current_class in range(len(map_classes)):

                if not include_background and current_class == 0:
                    continue

                y_true = np.where(original_es_mask == current_class, 1, 0).astype(np.int32)
                y_pred = np.where(pred_mask == current_class, 1, 0).astype(np.int32)
                class_str = class_to_cat[current_class]

                jc_score = jaccard_coef(y_true, y_pred)
                dc_score = dice_coef(y_true, y_pred)
                hd_score = secure_hd(y_true, y_pred)
                assd_score = secure_assd(y_true, y_pred)

                metrics[f'iou_{class_str}'].append(jc_score)
                metrics[f'dice_{class_str}'].append(dc_score)
                metrics[f'hd_{class_str}'].append(hd_score)
                metrics[f'assd_{class_str}'].append(assd_score)

            metrics[f'img_id'].append(img_id)
            metrics[f'phase'].append("ES")

    df = pd.DataFrame(metrics)
    # ## Get metrics by replacing infinite distance values with max value

    min_hausdorff_lv = df["hd_LV"].min()
    min_hausdorff_rv = df["hd_RV"].min()
    min_hausdorff_myo = df["hd_MYO"].min()

    min_assd_lv = df["assd_LV"].min()
    min_assd_rv = df["assd_RV"].min()
    min_assd_myo = df["assd_MYO"].min()

    max_hausdorff_lv = df["hd_LV"].max()
    max_hausdorff_rv = df["hd_RV"].max()
    max_hausdorff_myo = df["hd_MYO"].max()

    max_assd_lv = df["assd_LV"].max()
    max_assd_rv = df["assd_RV"].max()
    max_assd_myo = df["assd_MYO"].max()

    print(f"Mean IOU RV: {df['iou_RV'].mean()}")
    print(f"Mean IOU LV: {df['iou_LV'].mean()}")
    print(f"Mean IOU MYO: {df['iou_MYO'].mean()}")

    print("--------------")

    print(f"Mean DICE RV: {df['dice_RV'].mean()}")
    print(f"Mean DICE LV: {df['dice_LV'].mean()}")
    print(f"Mean DICE MYO: {df['dice_MYO'].mean()}")

    print("--------------")

    print(f"Mean Hausdorff RV: {df['hd_RV'].mean()}")
    print(f"Mean Hausdorff LV: {df['hd_LV'].mean()}")
    print(f"Mean Hausdorff MYO: {df['hd_MYO'].mean()}")

    print("--------------")

    print(f"Mean ASSD RV: {df['assd_RV'].mean()}")
    print(f"Mean ASSD LV: {df['assd_LV'].mean()}")
    print(f"Mean ASSD MYO: {df['assd_MYO'].mean()}")

    print(df.groupby("phase").mean())

    df.to_csv(os.path.join(preds_dir, f"results_{os.path.splitext(os.path.basename(check))[0]}.csv"), index=False)
