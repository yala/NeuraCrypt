import datetime
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from sandstone.utils.distributed_weighted_sampler import DistributedWeightedSampler
import hashlib
import random
from collections import defaultdict
from torch._six import container_abcs, string_classes, int_classes
import re

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts, MoleculeDatapoint or lists; found {}")

def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


INVALID_DATE_STR = "Date string not valid! Received {}, and got exception {}"
ISO_FORMAT = '%Y-%m-%dT%H:%M:%S'
def normalize_dictionary(dictionary):
    '''
    Normalizes counts in dictionary
    :dictionary: a python dict where each value is a count
    :returns: a python dict where each value is normalized to sum to 1
    '''
    num_samples = sum([dictionary[l] for l in dictionary])
    for label in dictionary:
        dictionary[label] = dictionary[label]*1. / num_samples
    return dictionary


def get_base_model_obj(model_dict, key):
    model = model_dict[key]
    if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    return model

def iso_str_to_datetime_obj(iso_string):
    '''
    Takes a string of format "YYYY-MM-DDTHH:MM:SS" and
    returns a corresponding datetime.datetime obj
    throws an exception if this can't be done.
    '''
    try:
        return datetime.datetime.strptime(iso_string, ISO_FORMAT)
    except Exception as e:
        raise Exception(INVALID_DATE_STR.format(iso_string, e))

def get_path_for_x(exam, args):
    exam_id = json.loads(exam)['id'] if 'cell_painter' in args.dataset else exam
    file_name = "{}.npy".format(exam_id)
    result_path_stem = args.results_path.split("/")[-1].split('.')[0]
    hiddens_dir = '{}_{}'.format(args.hiddens_dir, result_path_stem) if args.store_hiddens else args.hiddens_dir
    if not os.path.exists(hiddens_dir):
        os.makedirs(hiddens_dir)
    path = os.path.join(hiddens_dir, file_name)
    return path

def ignore_None_collate(batch):
    '''
    default_collate wrapper that creates batches only of not None values.
    Useful for cases when the dataset.__getitem__ can return None because of some
    exception and then we will want to exclude that sample from the batch.
    '''
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

def get_train_dataset_loader(args, train_data, batch_size):
    '''
        Given arg configuration, return appropriate torch.DataLoader
        for train_data and dev_data

        returns:
        train_data_loader: iterator that returns batches
        dev_data_loader: iterator that returns batches
    '''
    if args.class_bal:
        if args.distributed_backend == 'ddp':
            sampler = DistributedWeightedSampler(train_data, weights=train_data.weights, replacement=True, rank=args.global_rank, num_replicas = args.world_size)
        else:
            sampler = data.sampler.WeightedRandomSampler(
                weights=train_data.weights,
                num_samples=len(train_data),
                replacement=True)
    else:
        if args.distributed_backend == 'ddp':
            sampler = torch.utils.data.distributed.DistributedSampler(train_data, rank=args.global_rank, num_replicas = args.world_size)
        else:
            sampler = data.sampler.RandomSampler(train_data)

    train_data_loader = data.DataLoader(
                    train_data,
                    num_workers=args.num_workers,
                    sampler=sampler,
                    pin_memory=True,
                    batch_size=batch_size,
                    collate_fn=ignore_None_collate)

    return train_data_loader


def get_eval_dataset_loader(args, eval_data, batch_size, shuffle):
    drop_last = False
    if args.use_adv:
        drop_last = True

    if args.distributed_backend == 'ddp':
        sampler = torch.utils.data.distributed.DistributedSampler(eval_data, shuffle=shuffle, rank=args.global_rank, num_replicas = args.world_size)
    else:
        sampler = torch.utils.data.sampler.RandomSampler(eval_data) if shuffle else torch.utils.data.sampler.SequentialSampler(eval_data)
    data_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=batch_size,
        num_workers=args.num_workers,
        collate_fn=ignore_None_collate,
        pin_memory=True,
        drop_last=True,
        sampler = sampler)

    return data_loader

def save_as_numpy_arr(exam, x, args):
    '''
        Save x at path defined by name and args for later use.
    '''
    np.save(get_path_for_x(exam, args) , x)

def md5(key):
    '''
    returns a hashed with md5 string of the key
    '''
    return hashlib.md5(key.encode()).hexdigest()

def log(text, args):
    print(text)

def make_acc_to_x_test_dict(results_file):
    ACC_TO_X_KEYS = ['probs']
    exam_to_x = {}
    exams = results_file['test_stats']['exams']
    for key in ACC_TO_X_KEYS:
        exam_to_x['acc_to_{}'.format(key)] =  {exam: val for exam, val in zip (exams, results_file['test_stats'][key])}
    return exam_to_x


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
