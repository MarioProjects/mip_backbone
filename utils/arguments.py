import argparse
import json
import os


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(description='PAIP 2020 Challenge - CRC Prediction', formatter_class=SmartFormatter)

parser.add_argument("--gpu", type=str, default="0,1")
parser.add_argument("--seed", type=int, default=2020)
parser.add_argument('--output_dir', type=str, help='Where progress/checkpoints will be saved')

parser.add_argument(
    '--problem_type', type=str, default="", help='Deep Learning problem type.',
    choices=['classification', 'segmentation']
)

parser.add_argument('--epochs', type=int, default=150, help='Total number epochs for training')
parser.add_argument('--dataset', type=str, help='Which dataset use')
parser.add_argument('--defrost_epoch', type=int, default=-1, help='Number of epochs to defrost the model')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size for training')
parser.add_argument('--data_augmentation', type=str, help='Apply data augmentations at train time')
parser.add_argument('--img_size', type=int, default=224, help='Final img squared size')
parser.add_argument('--crop_size', type=int, default=224, help='Center crop squared size')

parser.add_argument('--normalization', type=str, required=True, help='Data normalization method')
parser.add_argument('--add_depth', action='store_true', help='If apply image transformation 1 to 3 channels or not')

parser.add_argument('--model_name', type=str, default='simple_unet', help='Model name for training')
parser.add_argument('--num_classes', type=int, default=1, help='Model output neurons')

# Accept a list of string metrics: train.py --metrics iou dice hauss
parser.add_argument('--metrics', '--names-list', nargs='+', default=[])

parser.add_argument('--generated_overlays', type=int, default=-1, help='Number of generate masks overlays')

parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer for training')
parser.add_argument('--scheduler', type=str, default="", help='Where is the model checkpoint saved')
parser.add_argument('--plateau_metric', type=str, default="", help='Metric name to set plateau')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--min_lr', type=float, default=0.0001, help='Minimun Learning rate')
parser.add_argument('--max_lr', type=float, default=0.01, help='Maximum Learning rate')
parser.add_argument(
    '--scheduler_steps', '--arg', nargs='+', type=int, help='Steps when steps and cyclic scheduler choosed'
)

parser.add_argument('--criterion', type=str, default='bce', help='Criterion for training')
parser.add_argument('--weights_criterion', type=str, default='default', help='Weights for each subcriterion')

parser.add_argument('--model_checkpoint', type=str, default="", help='If there is a model checkpoint to load')

parser.add_argument('--swa_freq', type=int, default=1, help='SWA Frequency')
parser.add_argument('--swa_start', type=int, default=-1, help='Epoch to start SWA and scheduler SWA_LR')
parser.add_argument('--swa_lr', type=float, default=0.05, help='SWA learning rate scheduler WA_LR')

parser.add_argument(
    '--mask_reshape_method', type=str, default="", help='How to reescale segmentation predictions.',
    choices=['padd', 'resize']
)

parser.add_argument('--selected_class', type=str, default="", help='If there is a model checkpoint to load')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

if args.output_dir == "":
    assert False, "Please set an output directory"

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

# https://stackoverflow.com/a/55114771
with open(os.path.join(args.output_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
