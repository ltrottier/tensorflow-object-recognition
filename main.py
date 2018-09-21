# global
import tensorflow as tf
import json
import argparse

# local
import dataset
import model
import train


# Parse arguments from opts file
parser = argparse.ArgumentParser()
parser.add_argument('optsfile', help='opts file generated with opts.py')
args = parser.parse_args()

with open(args.optsfile, 'r') as fid:
    opts = json.load(fid)

# dataset
dataset_name = opts['dataset_name']
dataset_dir = opts['dataset_dir']

# dataloader
batch_size = opts['dataloader_batch_size']
shuffle = opts['dataloader_shuffle']
num_workers = opts['dataloader_num_workers']
drop_last = opts['dataloader_drop_last']
augment = opts['dataloader_augment']

# optimizer
optimizer_name = opts['optim_name']
n_epochs = opts['optim_n_epochs']
lr_init = opts['optim_lr_init']
lr_decay = opts['optim_lr_decay']
lr_schedule = opts['optim_lr_schedule']
momentum = opts['optim_momentum']
nesterov = opts['optim_nesterov']
weight_decay = opts['optim_weight_decay']
loss_name = opts['optim_loss_name']

# network
network_name = opts['network_name']
network_args = opts['network_args']

# stats
stats_train_list = opts['stats_train_list']
stats_test_list = opts['stats_test_list']

# experiment
experiment_folder = opts['experiment_folder']


dataset_init = dataset.initialize(
    dataset_name,
    dataset_dir,
    batch_size,
    shuffle,
    num_workers,
    drop_last,
    augment)

input_data_ph = dataset_init[0]
target_data_ph = dataset_init[1]
batch_size_ph = dataset_init[2]
input_tensor = dataset_init[3]
target_tensor = dataset_init[4]
iterator_initializer_train = dataset_init[5]
iterator_initializer_test = dataset_init[6]
iterator_feed_dict_train = dataset_init[7]
iterator_feed_dict_test = dataset_init[8]

model_init = model.initialize(
    input_tensor,
    target_tensor,
    network_name,
    network_args,
    loss_name,
    stats_train_list,
    stats_test_list,
    optimizer_name,
    momentum,
    nesterov,
    weight_decay,
    lr_init,
    lr_decay,
    lr_schedule)

epoch_tensor = model_init[0]
train_begin = model_init[1]
train_step = model_init[2]
train_summary = model_init[3]
train_end = model_init[4]
test_begin = model_init[5]
test_step = model_init[6]
test_summary = model_init[7]
test_end = model_init[8]


train.loop(
    train_begin,
    train_step,
    train_summary,
    train_end,
    test_begin,
    test_step,
    test_summary,
    test_end,
    iterator_initializer_train,
    iterator_initializer_test,
    iterator_feed_dict_train,
    iterator_feed_dict_test,
    epoch_tensor,
    n_epochs,
    experiment_folder)

