#!/bin/bash

# Check if DRIVE data is available, if not download
if [ ! -d "data/DRIVE" ]
then
    echo "DRIVE data not found at 'data' directory. Downloading..."
    wget -nv --load-cookies /tmp/cookies.txt \
      "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt \
      --keep-session-cookies --no-check-certificate \
      'https://docs.google.com/uc?export=download&id=1kEMIClDQt34s48RdZBZbf4ET8jgbL5tS' \
      -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kEMIClDQt34s48RdZBZbf4ET8jgbL5tS" \
      -O drive_data.tar.gz && rm -rf /tmp/cookies.txt
    mkdir data
    tar -zxf drive_data.tar.gz  -C data/
    rm drive_data.tar.gz
    echo "Done!"
else
  echo "DRIVE data found at 'data' directory!"
fi

gpu="0"
dataset="DRIVE"
problem_type="segmentation"

# Available models:
#   -> resnet34_unet_scratch - resnet18_unet_scratch
#   -> small_segmentation_unet - small_segmentation_small_unet
#      small_segmentation_extrasmall_unet - small_segmentation_nano_unet
#   -> resnet18_pspnet_unet - resnet34_pspnet_unet
model="small_segmentation_extrasmall_unet"

img_size=224
crop_size=224

epochs=50
swa_start=40
defrost_epoch=-1
batch_size=32

# Available schedulers:
# constant - steps - plateau - one_cycle_lr (max_lr) - cyclic (min_lr, max_lr, scheduler_steps)
scheduler="constant"
lr=0.01
swa_lr=0.0256
# Available optimizers:
# adam - sgd - over9000
optimizer="adam"

# Available data augmentation policies:
# "none" - "random_crops" - "rotations" - "vflips" - "hflips" - "elastic_transform" - "grid_distortion" - "shift"
# "scale" - "optical_distortion" - "coarse_dropout" or "cutout" - "downscale" - "combination_da1"
data_augmentation="drive"

normalization="standardize"  # reescale - standardize
mask_reshape_method="resize"  # padd - resize

# Available criterions:
# bce - bce_dice - bce_dice_ac - bce_dice_border - bce_dice_border_ce
#criterion="bce_dice_border_ce"
#weights_criterion="0.5,0.2,0.2,0.2,0.5"
criterion="bce"
weights_criterion="1.0"

output_dir="$dataset/$model/$optimizer/${scheduler}_lr${lr}/${criterion}_weights${weights_criterion}"
output_dir="$output_dir/normalization_${normalization}/da${data_augmentation}"

python3 -u train.py --gpu $gpu --dataset $dataset --model_name $model --img_size $img_size --crop_size $crop_size \
--epochs $epochs --defrost_epoch $defrost_epoch --swa_start $swa_start --batch_size $batch_size \
--scheduler $scheduler --learning_rate $lr --swa_lr $swa_lr --optimizer $optimizer --criterion $criterion \
--normalization $normalization --weights_criterion $weights_criterion --data_augmentation $data_augmentation \
--output_dir "$output_dir" --metrics iou dice --problem_type $problem_type --mask_reshape_method $mask_reshape_method \
#--generate_overlays



