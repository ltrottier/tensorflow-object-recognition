# global
import tensorflow as tf
import json
import argparse

# local
import dataset
import model
import train


if False:
    parser = argparse.ArgumentParser()
    parser.add_argument('optsfile', help='opts file generated with opts.py')
    args = parser.parse_args()

    # load opts
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


#########################################################k

# dataset
dataset_name = 'cifar10'
dataset_dir = None
# dataloader
batch_size = 4
shuffle = True
num_workers = 4
drop_last = True
augment = True

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


# optimizer
optimizer_name = 'adam'
n_epochs = 10
lr_init = 0.001
lr_decay = 0.1
lr_schedule = [100]
momentum = 0.9
nesterov = True
weight_decay = 1e-4
loss_name = 'sparse_softmax_cross_entropy'

# network
network_name = 'resnet'
network_args = [10, 18]

# stats
stats_train_list = ['error_rate', 'loss_average', 'input_image_visualization']
stats_test_list = ['error_rate', 'loss_average', 'input_image_visualization']

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


# experiment
experiment_folder = 'results/debug'

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

