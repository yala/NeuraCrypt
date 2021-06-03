import copy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sandstone.datasets.factory as dataset_factory
import sandstone.augmentations.factory as transformer_factory
from sandstone.utils.parsing import parse_augmentations
from sandstone.utils.generic import normalize_dictionary, get_train_dataset_loader

import sys
from os.path import dirname, realpath
import pdb

sys.path.append(dirname(dirname(realpath(__file__))))

def modify_args(args):
    # Set transformer to resize image img_size before computing stats
    # to improve computation speed and memory overhead
    args.cache_path = None

def get_dataset_stats(args):
    args = copy.deepcopy(args)
    modify_args(args)

    augmentations = transformer_factory.get_augmentations(args.image_augmentations, [], args)

    train, _, _ = dataset_factory.get_dataset(args, augmentations, [])

    data_loader = get_train_dataset_loader(args, train, args.batch_size)

    means, stds = {i:[] for i in range(args.num_chan)}, {i:[] for i in range(args.num_chan)}
    mins, maxs = {i:[] for i in range(args.num_chan)}, {i:[] for i in range(args.num_chan)}

    indx = 1
    for batch in tqdm(data_loader):
        tensor = batch['x']
        for channel in range(args.num_chan):
            tensor_chan = tensor[:, channel]
            means[channel].append(torch.mean(tensor_chan))
            stds[channel].append(torch.std(tensor_chan))
            mins[channel].append(torch.min(tensor_chan))
            maxs[channel].append(torch.max(tensor_chan))
        if indx % (len(data_loader)//20) == 0:
            _means = [torch.mean(torch.Tensor(means[channel])).item() for channel in range(args.num_chan)]
            _stds = [torch.mean(torch.Tensor(stds[channel])).item() for channel in range(args.num_chan)]
            _max = [torch.max(torch.Tensor(maxs[channel])).item() for channel in range(args.num_chan)]
            _min = [torch.min(torch.Tensor(mins[channel])).item() for channel in range(args.num_chan)]
            print('for indx={}\t mean={}\t std={}\t max={} \t min={}'.format(indx, _means, _stds, _max, _min))
        indx += 1
    means = [torch.mean(torch.Tensor(means[channel])) for channel in range(args.num_chan)]
    stds = [torch.mean(torch.Tensor(stds[channel])) for channel in range(args.num_chan)]

    return means, stds
