import comet_ml
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from os.path import dirname, realpath
import sys
import git
sys.path.append(dirname(dirname(realpath(__file__))))
import torch
import torch.distributed as dist
import sandstone.datasets.factory as dataset_factory
import sandstone.models.factory as model_factory
import sandstone.augmentations.factory as augmentation_factory
import sandstone.utils.parsing as parsing
from sandstone.utils.generic import get_train_dataset_loader, get_eval_dataset_loader, get_nearest_neighbor_in_encoded_data
import warnings
from sandstone.utils.dataset_stats import get_dataset_stats
import pytorch_lightning as pl
import sandstone.learn.lightning.factory as lightning
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import _logger as log
import tqdm
import numpy as np
import json


#Constants
DATE_FORMAT_STR = "%Y-%m-%d:%H-%M-%S"

@torch.no_grad()
def main(args):
    repo = git.Repo(search_parent_directories=True)
    commit  = repo.head.object
    args.commit = commit.hexsha
    result_path_stem = args.results_path.split("/")[-1].split('.')[0]
    log.info("Sandstone main running from commit: \n\n{}\n{}author: {}, date: {}".format(
        commit.hexsha, commit.message, commit.author, commit.committed_date))


    log.info("\nLoading data-augmentation scheme...")
    augmentations = augmentation_factory.get_augmentations(
        args.image_augmentations, args.tensor_augmentations, args)
    test_augmentations = augmentation_factory.get_augmentations(
        args.test_image_augmentations, args.test_tensor_augmentations, args)
    # Load dataset and add dataset specific information to args

    evaluate_parallel_data_challenge(args)
    evaluate_real_world_challenge(args, augmentations, test_augmentations)


def evaluate_parallel_data_challenge(args):
    print("### Evaluate parallel data Challenge submission \n")

    ## Save real paths, args, and model for future eval
    preds = json.load(open(os.path.join(args.submission_dir, 'parallel_data_predictions.json' ), 'r'))
    golds = json.load(open(os.path.join(args.encoded_data_dir, 'path_dict.json'), 'r'))

    matching = []
    for k,v in preds.items():
        if k in golds:
            matching.append(v == golds[k])
        else:
            print("warning, {} not in golds..")
    print("Matching Accuracy {} ({}/{}) \n \n ##".format(np.mean(matching), np.sum(matching), len(matching)))

def evaluate_real_world_challenge(args, augmentations, test_augmentations):
    print("### Prepare real world (no-parallel data) Challenge submission")
    dataset_name = 'mimic_cxr_edema'

    npy_path_to_orig_path_submitted  = json.load(open(os.path.join(args.submission_dir, 'out_of_domain_npy_path_to_orig_path_dict.json' ), 'r'))
    orig_paths_submitted = set( npy_path_to_orig_path_submitted.values() )
    _, _, test_data = dataset_factory.get_dataset_by_name(dataset_name, args, augmentations, test_augmentations)

    test_data.dataset = [d for d in test_data.dataset if d['path'] in orig_paths_submitted]

    loader = get_eval_dataset_loader(args, test_data, args.batch_size, True)

    path_to_pred_npy_path = {}

    matching = []
    for batch in tqdm.tqdm(loader):

        z = batch['z'].mean(dim=1).cpu().numpy()
        B = z.shape[0]
        for j in range(B):
            true_path = batch['path'][j]
            true_npy_path = batch['z_path'][j]

            predicted_npy_path = get_nearest_neighbor_in_encoded_data(z[j], os.path.join(args.submission_dir, 'mimic'), reduce_mean=False)
            submitted_path = npy_path_to_orig_path_submitted[predicted_npy_path]
            matching.append( submitted_path == true_path )

    print("Matching Accuracy {} ({}/{})".format(np.mean(matching), np.sum(matching), len(matching)))


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = parsing.parse_args()
    main(args)
