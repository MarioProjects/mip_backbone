#!/bin/bash

# Check if ACDC 2017 data is available, if not download
if [ ! -d "data/AC17" ]
then
    echo "AC17 data not found at 'data' directory. Downloading..."
    curl -O -J https://nextcloud.maparla.duckdns.org/s/cAENNxDn4E4rm7z/download
    mkdir -p data
    tar -zxf acdc_2017.tar.gz  -C data/
    rm acdc_2017.tar.gz
    echo "Done!"
    [ "$1" == "only_data" ] && exit
else
  echo "AC17 already downloaded!"
  # Only download the data argument ./tests/segmentation/acdc172d.sh only_data
  [ "$1" == "only_data" ] && exit
fi


gpu="0,1"
dataset="ACDC172D"
problem_type="segmentation"

acdc_train_patients=10

# Available models:
#   -> resnet34_unet_scratch - resnet18_unet_scratch
#   -> small_segmentation_unet - small_segmentation_small_unet
#      small_segmentation_extrasmall_unet - small_segmentation_nano_unet
#   -> resnet18_pspnet_unet - resnet34_pspnet_unet
model="resnet34_unet_scratch"

img_size=224
crop_size=224
batch_size=4

epochs=120
swa_start=80
defrost_epoch=7

# Available schedulers:
# constant - steps - plateau - one_cycle_lr (max_lr) - cyclic (min_lr, max_lr, scheduler_steps)
scheduler="steps"
lr=0.0001
swa_lr=0.00256
# Available optimizers:
# adam - sgd - over9000
optimizer="adam"

# Available data augmentation policies:
# "none" - "random_crops" - "rotations" - "vflips" - "hflips" - "elastic_transform" - "grid_distortion" - "shift"
# "scale" - "optical_distortion" - "coarse_dropout" or "cutout" - "downscale"
data_augmentation="acdc172d"

normalization="standardize"  # reescale - standardize
mask_reshape_method="padd"  # padd - resize

generated_overlays=0

# Available criterions:
# bce - bce_dice - bce_dice_ac - bce_dice_border - bce_dice_border_ce
#criterion="bce_dice_border_ce"
#weights_criterion="0.5,0.2,0.2,0.2,0.5"
criterion="bce_dice"
weights_criterion="0.4, 0.5, 0.1"

output_dir="results/$dataset/${acdc_train_patients}_patients/$model/$optimizer/${scheduler}_lr${lr}"
output_dir="$output_dir/${criterion}_weights${weights_criterion}/normalization_${normalization}/da${data_augmentation}"

python3 -u train.py --gpu $gpu --dataset $dataset --model_name $model --img_size $img_size --crop_size $crop_size \
--epochs $epochs --swa_start $swa_start --batch_size $batch_size --defrost_epoch $defrost_epoch \
--scheduler $scheduler --learning_rate $lr --swa_lr $swa_lr --optimizer $optimizer --criterion $criterion \
--normalization $normalization --weights_criterion "$weights_criterion" --data_augmentation $data_augmentation \
--output_dir "$output_dir" --metrics iou dice --problem_type $problem_type --mask_reshape_method $mask_reshape_method \
--scheduler_steps 45 65 --generated_overlays $generated_overlays --add_depth --acdc_train_patients $acdc_train_patients



