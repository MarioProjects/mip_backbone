#!/bin/bash

# Check if DRIVE data is available, if not download
if [ ! -d "data/SIMEPU_Segmentation" ]
then
    echo "SIMEPU Segmentation data not found at 'data' directory. Downloading..."
    wget -nv --load-cookies /tmp/cookies.txt \
      "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt \
      --keep-session-cookies --no-check-certificate \
      'https://docs.google.com/uc?export=download&id=1tPw_phod7s1T_7cvVhrsHH25l47mifOI' \
      -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tPw_phod7s1T_7cvVhrsHH25l47mifOI" \
      -O simepu_segmentation.tar.gz && rm -rf /tmp/cookies.txt
    mkdir data
    tar -zxf simepu_segmentation.tar.gz  -C data/
    rm simepu_segmentation.tar.gz
    echo "Done!"
else
  echo "SIMEPU Segmentation data found at 'data' directory!"
fi

gpu="0,1"
dataset="SIMEPUSegmentation"
problem_type="segmentation"

# Available models:
#   -> resnet34_unet_scratch - resnet18_unet_scratch
#   -> small_segmentation_unet - small_segmentation_small_unet
#      small_segmentation_extrasmall_unet - small_segmentation_nano_unet
#   -> resnet18_pspnet_unet - resnet34_pspnet_unet
model="small_segmentation_extra_small_unet"

# Available classes: Grietas longitudinales - Grietas transversales - Huecos - Parcheo
selected_class="Parcheo"

img_size=512
crop_size=512
batch_size=28

epochs=100
swa_start=-1

# Available schedulers:
# constant - steps - plateau - one_cycle_lr (max_lr) - cyclic (min_lr, max_lr, scheduler_steps)
scheduler="steps"
lr=0.01
swa_lr=0.0256
# Available optimizers:
# adam - sgd - over9000
optimizer="adam"

# Available data augmentation policies:
# "none" - "random_crops" - "rotations" - "vflips" - "hflips" - "elastic_transform" - "grid_distortion" - "shift"
# "scale" - "optical_distortion" - "coarse_dropout" or "cutout" - "downscale" - "combination_da1"
data_augmentation="simepu_segmentation"

normalization="reescale"  # reescale - standardize
mask_reshape_method="resize"  # padd - resize

generated_overlays=10

# Available criterions:
# bce - bce_dice - bce_dice_ac - bce_dice_border - bce_dice_border_ce
#criterion="bce_dice_border_ce"
#weights_criterion="0.5,0.2,0.2,0.2,0.5"
criterion="bce"
weights_criterion="1.0"

output_dir="results/$dataset/$model/$optimizer/${scheduler}_lr${lr}/${criterion}_weights${weights_criterion}"
output_dir="$output_dir/normalization_${normalization}/da${data_augmentation}"

python3 -u train.py --gpu $gpu --dataset $dataset --model_name $model --img_size $img_size --crop_size $crop_size \
--epochs $epochs --swa_start $swa_start --batch_size $batch_size --selected_class $selected_class \
--scheduler $scheduler --learning_rate $lr --swa_lr $swa_lr --optimizer $optimizer --criterion $criterion \
--normalization $normalization --weights_criterion $weights_criterion --data_augmentation $data_augmentation \
--output_dir "$output_dir" --metrics iou --problem_type $problem_type --mask_reshape_method $mask_reshape_method \
--scheduler_steps 50 80 --generated_overlays $generated_overlays



