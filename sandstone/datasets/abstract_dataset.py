import numpy as np
import pickle
from abc import ABCMeta, abstractmethod
import torch
from torch.utils import data
import os
import warnings
import json
import traceback
from collections import Counter
from sandstone.datasets.loader.factory import get_input_loader
from scipy.stats import entropy
from sandstone.utils.generic import log
from sandstone.utils.generic import get_path_for_x
import pdb

METAFILE_NOTFOUND_ERR = "Metadata file {} could not be parsed! Exception: {}!"
LOAD_FAIL_MSG = "Failed to load image: {}\nException: {}"

DEBUG_SIZE=1000

DATASET_ITEM_KEYS = ['ssn', 'pid', 'exam', 'source', 'path']

class Abstract_Dataset(data.Dataset):
    """
    Abstract Object for all Datasets. All datasets have some metadata
    property associated with them, a create_dataset method, a task, and a check
    label and get label function.
    """
    __metaclass__ = ABCMeta

    def __init__(self, args, augmentations, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in an image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        '''
        super(Abstract_Dataset, self).__init__()
        self.split_group = split_group
        self.args = args
        self.input_loader = get_input_loader(args.cache_path, augmentations, args)
        if hasattr(self, 'METADATA_FILENAME') and self.METADATA_FILENAME is not None:
            args.metadata_path = os.path.join(args.metadata_dir,
                                              self.METADATA_FILENAME)

            try:
                self.metadata_json = json.load(open(args.metadata_path, 'r'))
            except Exception as e:
                raise Exception(METAFILE_NOTFOUND_ERR.format(args.metadata_path, e))

            if args.debug and isinstance(self.metadata_json,list):
                self.metadata_json = self.metadata_json[:DEBUG_SIZE]

        self.path_to_hidden_dict = {}
        self.dataset = self.create_dataset(split_group, args.img_dir)
        if len(self.dataset) == 0:
            return
        if split_group == 'train' and self.args.data_fraction < 1.0:
            self.dataset = np.random.choice(self.dataset, int(len(self.dataset)*self.args.data_fraction), replace=False)
        try:
            self.add_device_to_dataset()
        except:
            log("Could not add device information to dataset", args)
        for d in self.dataset:
            if 'exam' in d and 'year' in d:
                args.exam_to_year_dict[d['exam']] = d['year']
        log(self.get_summary_statement(self.dataset, split_group), args)

        if 'dist_key' in self.dataset[0]:
            dist_key = 'dist_key'
        else:
            dist_key = 'y'

        label_dist = [d[dist_key] for d in self.dataset]
        label_counts = Counter(label_dist)
        weight_per_label = 1./ len(label_counts)
        label_weights = {
            label: weight_per_label/count for label, count in label_counts.items()
            }
        if args.class_bal and args.num_classes < 10:
            log("Class counts are: {}".format(label_counts), args)
            log("Label weights are {}".format(label_weights), args)
        self.weights = [ label_weights[d[dist_key]] for d in self.dataset]




    @property
    @abstractmethod
    def task(self):
        pass

    @property
    @abstractmethod
    def METADATA_FILENAME(self):
        pass

    @property
    def is_ct_dataset(self):
        return False

    @abstractmethod
    def check_label(self, row):
        '''
        Return True if the row contains a valid label for the task
        :row: - metadata row
        '''
        pass

    @abstractmethod
    def get_label(self, row):
        '''
        Get task specific label for a given metadata row
        :row: - metadata row with contains label information
        '''
        pass

    def get_summary_statement(self, dataset, split_group):
        '''
        Return summary statement
        '''
        return ""

    @abstractmethod
    def create_dataset(self, split_group, img_dir):
        """
        Creating the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.

        """
        pass


    @staticmethod
    def set_args(args):
        """Sets any args particular to the dataset."""
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        try:
            if self.args.multi_image:
                x = self.input_loader.get_images(sample['paths'], sample['additionals'], sample)

            else:
                if ( ('additional' in sample) and (sample['additional'] is None) ) or ('additional' not in sample):
                    sample['additional'] = {}
                x = self.input_loader.get_image(sample['path'], sample['additional'], sample)

            item = {
                'x': x,
                'y': sample['y']
                }

            for key in DATASET_ITEM_KEYS:
                if key in sample:
                    item[key] = sample[key]

            return item
        except Exception:
            path_key =  'paths' if  self.args.multi_image  else 'path'
            warnings.warn(LOAD_FAIL_MSG.format(sample[path_key], traceback.print_exc()))
