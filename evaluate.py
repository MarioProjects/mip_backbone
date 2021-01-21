#!/usr/bin/env python
# coding: utf-8

# ---- My utils ----
from models import model_selector
from utils.arguments import *
from utils.data_augmentation import data_augmentation_selector
from utils.datasets import dataset_selector
from utils.neural import *

_, _, val_aug = data_augmentation_selector(
    args.data_augmentation, args.img_size, args.crop_size, args.mask_reshape_method
)
test_loader = dataset_selector(_, _, val_aug, args, is_test=True)

model = model_selector(
    args.problem_type, args.model_name, test_loader.dataset.num_classes, from_swa=args.swa_checkpoint,
    in_channels=test_loader.dataset.img_channels, devices=args.gpu, checkpoint=args.model_checkpoint
)

test_metrics = MetricsAccumulator(
    args.problem_type, args.metrics, test_loader.dataset.num_classes, average="mean",
    include_background=test_loader.dataset.include_background, mask_reshape_method=args.mask_reshape_method
)

test_metrics, cases_ids = test_step(
    test_loader, model, test_metrics, generated_overlays=args.generated_overlays,
    overlays_path=f"{args.output_dir}/overlays/test_evaluation"
)

test_metrics.save_progress_cases(cases_ids, args.output_dir, identifier="test_metrics")
print("\nResults:")
test_metrics.update()
test_metrics.report_best()

if args.notify:
    slack_message(message=f"{args.dataset.upper()} evaluation experiments finished!", channel="experiments")
