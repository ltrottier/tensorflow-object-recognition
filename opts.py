import os
import json
from argparse import ArgumentParser

# parse arguments
parser = ArgumentParser()

# dataset
parser.add_argument('--dataset-name', default='cifar10')
parser.add_argument('--dataset-dir', default='')

# dataloader
parser.add_argument('--dataloader-batch-size', type=int, default=32)
parser.add_argument('--dataloader-shuffle', dest='dataloader_shuffle', action='store_true')
parser.add_argument('--no-dataloader-shuffle', dest='dataloader_shuffle', action='store_false')
parser.set_defaults(dataloader_shuffle=True)
parser.add_argument('--dataloader-num-workers', type=int, default=2)
parser.add_argument('--dataloader-drop-last', dest='dataloader_drop_last', action='store_true')
parser.add_argument('--no-dataloader-drop-last', dest='dataloader_drop_last', action='store_false')
parser.set_defaults(dataloader_drop_last=True)
parser.add_argument('--dataloader-augment', dest='dataloader_augment', action='store_true')
parser.add_argument('--no-dataloader-augment', dest='dataloader_augment', action='store_false')
parser.set_defaults(dataloader_augment=True)

# optim
parser.add_argument('--optim-name', default='sgd')
parser.add_argument('--optim-n-epochs', type=int, default=300)
parser.add_argument('--optim-lr-init', type=float, default=0.1)
parser.add_argument('--optim-lr-decay', type=float, default=0.2)
parser.add_argument('--optim-lr-schedule', nargs='+', default=[100, 180, 240, 280], type=int)
parser.add_argument('--optim-momentum', type=float, default=0.9)
parser.add_argument('--optim-nesterov', dest='optim_nesterov', action='store_true')
parser.add_argument('--no-optim-nesterov', dest='optim_nesterov', action='store_false')
parser.set_defaults(optim_nesterov=True)
parser.add_argument('--optim-weight-decay', type=float, default=1e-4)
parser.add_argument('--optim-loss-name', default='sparse_softmax_cross_entropy')

# network
parser.add_argument('--network-name', default='resnet')
parser.add_argument('--network-args', nargs='+', default=[10, 18], type=int)

# stats
parser.add_argument('--stats-train-list', nargs='+',
                    default=['error_rate', 'loss_average', 'input_image_visualization'])
parser.add_argument('--stats-test-list', nargs='+',
                    default=['error_rate', 'loss_average', 'input_image_visualization'])

# experiment
parser.add_argument('--experiment-folder', default='results/exp1')

# parse
args = parser.parse_args()
opts = vars(args)

# create experiment folder
experiment_folder = opts['experiment_folder']
os.makedirs(experiment_folder)

# save opts
opts_filename = os.path.join(experiment_folder, 'opts.txt')
with open(opts_filename, 'w') as fid:
    json.dump(opts, fid)

