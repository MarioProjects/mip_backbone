#!/usr/bin/env python
# coding: utf-8

from torch.optim.swa_utils import AveragedModel

# ---- My utils ----
from models import model_selector
from utils.arguments import *
from utils.data_augmentation import data_augmentation_selector
from utils.datasets import dataset_selector
from utils.logging import log_epoch, build_header
from utils.neural import *

train_aug, train_aug_img, val_aug = data_augmentation_selector(
    args.data_augmentation, args.img_size, args.crop_size, args.mask_reshape_method
)
train_loader, val_loader = dataset_selector(train_aug, train_aug_img, val_aug, args)

model = model_selector(
    args.problem_type, args.model_name, train_loader.dataset.num_classes,
    in_channels=train_loader.dataset.img_channels, devices=args.gpu, checkpoint=args.model_checkpoint
)

swa_model = None

criterion, weights_criterion, multiclass_criterion = get_criterion(args.criterion, args.weights_criterion)
optimizer = get_optimizer(args.optimizer, model, lr=args.learning_rate)

scheduler = get_scheduler(
    args.scheduler, optimizer, epochs=args.epochs,
    min_lr=args.min_lr, max_lr=args.max_lr, scheduler_steps=args.scheduler_steps
)

swa_scheduler = get_scheduler("swa", optimizer, max_lr=args.swa_lr) if args.swa_start != -1 else None

train_metrics = MetricsAccumulator(
    args.problem_type, args.metrics, train_loader.dataset.num_classes, average="mean",
    include_background=train_loader.dataset.include_background, mask_reshape_method=args.mask_reshape_method
)
val_metrics = MetricsAccumulator(
    args.problem_type, args.metrics, train_loader.dataset.num_classes, average="mean",
    include_background=train_loader.dataset.include_background, mask_reshape_method=args.mask_reshape_method
)

header, defrosted = build_header(train_loader.dataset.class_to_cat, args.metrics, display=True), False
for current_epoch in range(args.epochs):

    defrosted = check_defrost(model, defrosted, current_epoch, args.defrost_epoch)

    train_metrics = train_step(
        train_loader, model, criterion, weights_criterion, multiclass_criterion, optimizer, train_metrics
    )

    val_metrics = val_step(
        val_loader, model, val_metrics, generated_overlays=args.generated_overlays,
        overlays_path=f"{args.output_dir}/overlays/epoch_{current_epoch}"
    )

    current_lr = get_current_lr(optimizer)
    log_epoch((current_epoch + 1), current_lr, train_metrics, val_metrics, header)

    val_metrics.save_progress(args.output_dir, identifier="validation_metrics")
    train_metrics.save_progress(args.output_dir, identifier="train_metrics")

    if args.swa_start != -1 and (current_epoch + 1) >= args.swa_start:
        if swa_model is None:
            print("\n------------------------------- START SWA -------------------------------\n")
            swa_model = AveragedModel(model)
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        # Only save checkpoints when not applying SWA -> only want save last model using SWA
        create_checkpoint(val_metrics, model, args.model_name, args.output_dir)
        scheduler_step(optimizer, scheduler, val_metrics, args)

print("\nBest Validation Results:")
val_metrics.report_best()

finish_swa(swa_model, train_loader, val_loader, args)

if args.notify:
    slack_message(message=f"{args.dataset.upper()} experiments finished!", channel="experiments")
