import os
from collections import Counter
import torch
from sandstone.datasets.factory import RegisterDataset
from sandstone.datasets.abstract_dataset import Abstract_Dataset
import sandstone.utils
import tqdm
from random import shuffle
import copy
import numpy as np
import datetime
import pdb

MIMIC_METADATA_PATH = "data/mimic.json"
STANFORD_METADATA_PATH = "data/chexpert.json"
ALL_METADATA_PATH = "data/combined.json"

SUMMARY_MSG = "Contructed CXR {} {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"

class Abstract_Cxr(Abstract_Dataset):
    '''
        Working dataset for suvival analysis.
    '''
    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.
        """
        dataset = []
        for row in tqdm.tqdm(self.metadata_json):
            ssn, split, exam = row['pid'], row['split_group'], row['exam']
            split = 'dev' if row['split_group'] == 'validate' else row['split_group']

            ## Use challenge splits for attack mode. Only implemented for CheXpert (i.e Stanford)
            if self.args.lightning_name == "adversarial_attack" and 'challenge_split' in row:
                if split_group in ['dev','train']:
                    if not row['challenge_split'] == 'public':
                        continue
                else:
                    assert split_group == 'test'
                    if not row['challenge_split'] == 'private-encoded':
                        continue
            else:
                if split != split_group:
                    continue

            if self.check_label(row):
                label = self.get_label(row)
                source = 0
                if self.args.rlc_private_multi_host or self.args.use_adv:
                    if not 'source' in row:
                        row['source'] = 'mimic' if 'MIMIC' in self.METADATA_FILENAME else 'stanford'
                    assert row['source'] in ['mimic', 'stanford']
                    source = 1 if row['source'] == 'mimic' else 0
                dataset.append({
                    'path': row['path'],
                    'y': label,
                    'additional': {},
                    'exam': exam,
                    'source': source,
                    'ssn': ssn
                })

        if self.args.use_shuffle_pairs:
            shuffle_dataset = copy.deepcopy(dataset)
            np.random.shuffle(shuffle_dataset)

            for source, target in zip(dataset, shuffle_dataset):
                source['paths'] = [source['path'], target['path']]
                source['additionals'] = []
                del source['additional']
                del source['path']
            dataset = dataset[:128]

        return dataset

    def get_summary_statement(self, dataset, split_group):
        class_balance = Counter([d['y'] for d in dataset])
        exams = set([d['exam'] for d in dataset])
        patients = set([d['ssn'] for d in dataset])
        statement = SUMMARY_MSG.format(self.task, split_group, len(dataset), len(exams), len(patients), class_balance)
        return statement

    def get_label(self, row):
        return row['label_dict'][self.task] == "1.0"


    def check_label(self, row):
        if self.args.lightning_name == "adversarial_attack":
            return True
        return row['label_dict'][self.task] in ["1.0", "0.0"] or ( 'No Finding' in row['label_dict'] and row['label_dict']['No Finding'] == "1.0")

    @staticmethod
    def set_args(args):
        args.num_classes = 2
        args.num_chan = 1
        args.num_hospitals = 2
        args.img_size = (256, 256)
        args.img_mean = [43.9]
        args.img_std = [63.2]
        args.input_loader_name = 'default_image_loader'
        args.image_augmentations = ["scale_2d"]
        args.tensor_augmentations = ["normalize_2d"]
        args.test_image_augmentations = ["scale_2d"]
        args.test_tensor_augmentations = ["normalize_2d"]

        if args.use_shuffle_pairs:
            args.multi_image = True
            args.num_images = 2

class Abstract_Mimic_Cxr(Abstract_Cxr):
    @property
    def METADATA_FILENAME(self):
        return MIMIC_METADATA_PATH

class Abstract_Stanford_Cxr(Abstract_Cxr):
    @property
    def METADATA_FILENAME(self):
        return STANFORD_METADATA_PATH

class Abstract_Combined_Cxr(Abstract_Cxr):
    @property
    def METADATA_FILENAME(self):
        return ALL_METADATA_PATH


## Mimic Datasets
@RegisterDataset("mimic_pneumothorax")
class Mimic_Cxr_Pneumothorax(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return "Pneumothorax"

@RegisterDataset("mimic_cxr_edema")
class Mimic_Cxr_Edema(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return 'Edema'

@RegisterDataset("mimic_cxr_consolidation")
class Mimic_Cxr_Consolidation(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return 'Consolidation'

@RegisterDataset("mimic_cxr_cardiomegaly")
class Mimic_Cxr_Cardiomegaly(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return 'Cardiomegaly'

@RegisterDataset("mimic_cxr_atelectasis")
class Mimic_Cxr_Atelectasis(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return 'Atelectasis'


## Stanford Datasets
@RegisterDataset("stanford_pneumothorax")
class Stanford_Cxr_Pneumothorax(Abstract_Stanford_Cxr):
    @property
    def task(self):
        return "Pneumothorax"

@RegisterDataset("stanford_cxr_edema")
class Stanford_Cxr_Edema(Abstract_Stanford_Cxr):
    @property
    def task(self):
        return 'Edema'

@RegisterDataset("stanford_cxr_consolidation")
class Stanford_Cxr_Consolidation(Abstract_Stanford_Cxr):
    @property
    def task(self):
        return 'Consolidation'

@RegisterDataset("stanford_cxr_cardiomegaly")
class Stanford_Cxr_Cardiomegaly(Abstract_Stanford_Cxr):
    @property
    def task(self):
        return 'Cardiomegaly'

@RegisterDataset("stanford_cxr_atelectasis")
class Stanford_Cxr_Atelectasis(Abstract_Stanford_Cxr):
    @property
    def task(self):
        return 'Atelectasis'


## Combined Datasets
@RegisterDataset("combined_pneumothorax")
class Combined_Cxr_Pneumothorax(Abstract_Combined_Cxr):
    @property
    def task(self):
        return "Pneumothorax"

@RegisterDataset("combined_cxr_edema")
class Combined_Cxr_Edema(Abstract_Combined_Cxr):
    @property
    def task(self):
        return 'Edema'

@RegisterDataset("combined_cxr_consolidation")
class Combined_Cxr_Consolidation(Abstract_Combined_Cxr):
    @property
    def task(self):
        return 'Consolidation'

@RegisterDataset("combined_cxr_cardiomegaly")
class Combined_Cxr_Cardiomegaly(Abstract_Combined_Cxr):
    @property
    def task(self):
        return 'Cardiomegaly'

@RegisterDataset("combined_cxr_atelectasis")
class Combined_Cxr_Atelectasis(Abstract_Combined_Cxr):
    @property
    def task(self):
        return 'Atelectasis'
