#!/bin/bash
# Only download the data argument ./tests/segmentation/lvsc2d.sh only_data
# Check if LVSC data is available, if not download
if [ ! -d "data/LVSC" ]
then
    echo "LVSC data not found at 'data' directory. Downloading..."
    curl -O -J https://nextcloud.maparla.duckdns.org/s/cRiw9Zdi8JkZJNN/download
    mkdir -p data
    tar -zxf lv_lvsc.tar.gz  -C data/
    rm lv_lvsc.tar.gz
    echo "Done!"
    [ "$1" == "only_data" ] && exit
else
  echo "LVSC already downloaded!"
  [ "$1" == "only_data" ] && exit
fi

for lv_train_patients in 5 10 25
do

for model in "resnet34_unet_scratch_scse_hypercols" "resnet34_unet_imagenet_encoder_scse_hypercols"
do

gpu="0,1"
dataset="LVSC2D"
problem_type="segmentation"

# Available models:
#   -> resnet34_unet_scratch - resnet18_unet_scratch
#   -> small_segmentation_unet - small_segmentation_small_unet
#      small_segmentation_extrasmall_unet - small_segmentation_nano_unet
#   -> resnet18_pspnet_unet - resnet34_pspnet_unet
#model="resnet34_unet_imagenet_encoder"

#lv_train_patients=100

img_size=224
crop_size=224
batch_size=32

epochs=135
swa_start=90
defrost_epoch=8

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
data_augmentation="lvsc2d"

normalization="standardize"  # reescale - standardize
mask_reshape_method="padd"  # padd - resize

generated_overlays=10

# Available criterions:
# bce - bce_dice - bce_dice_ac - bce_dice_border - bce_dice_border_ce
#criterion="bce_dice_border_ce"
#weights_criterion="0.5,0.2,0.2,0.2,0.5"
criterion="bce_dice"
weights_criterion="0.4, 0.5, 0.1"

output_dir="results/$dataset/${lv_train_patients}_patients/$model/$optimizer/${scheduler}_lr${lr}"
output_dir="$output_dir/${criterion}_weights${weights_criterion}/normalization_${normalization}/da${data_augmentation}"

python3 -u train.py --gpu $gpu --dataset $dataset --model_name $model --img_size $img_size --crop_size $crop_size \
--epochs $epochs --swa_start $swa_start --batch_size $batch_size --defrost_epoch $defrost_epoch \
--scheduler $scheduler --learning_rate $lr --swa_lr $swa_lr --optimizer $optimizer --criterion $criterion \
--normalization $normalization --weights_criterion "$weights_criterion" --data_augmentation $data_augmentation \
--output_dir "$output_dir" --metrics iou dice --problem_type $problem_type --mask_reshape_method $mask_reshape_method \
--scheduler_steps 45 65 --generated_overlays $generated_overlays --add_depth --lv_train_patients $lv_train_patients

model_checkpoint="$output_dir/model_${model}_${epochs-swa_start}epochs_swalr${swa_lr}.pt"
python3 -u evaluate.py --gpu $gpu --dataset $dataset --model_name $model --img_size $img_size --crop_size $crop_size \
--swa_checkpoint --batch_size $batch_size --normalization $normalization --output_dir "$output_dir" --metrics iou dice \
--problem_type $problem_type --mask_reshape_method $mask_reshape_method \
--generated_overlays $generated_overlays --add_depth --model_checkpoint "$model_checkpoint"

done

done