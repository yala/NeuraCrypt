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
    log.info("\nLoading data...")

    args.lightning_name = 'adversarial_attack'

    model = lightning.get_lightning_model(args)
    log.info("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if attr not in ['optimizer_state']:
            log.info("\t{}={}".format(attr.upper(), value))

    save_path = args.results_path
    
    print("Model")
    print(model.attack_encoder)
    model = model.cuda()

    parallel_data_challenge(args, augmentations, test_augmentations, model)
    real_world_challenge(args, augmentations, test_augmentations, model)

    args.model_path = os.path.join(args.submission_dir, 'model.p')
    pickle.dump(vars(args), open(os.path.join(args.submission_dir, 'args.p' ), 'wb'))
    torch.save(model, args.model_path)

def parallel_data_challenge(args, augmentations, test_augmentations, model):
    if os.path.exists( os.path.join(args.submission_dir, 'parallel_data_predictions.json' )):
        print("Parallel data submission already created. Skipping creation..")
        return
    print("### Prepare parallel data Challenge submission")
    dataset_name = 'stanford_cxr_edema'
    train_data, dev_data, test_data = dataset_factory.get_dataset_by_name(dataset_name, args, augmentations, test_augmentations)

    train_loader = get_train_dataset_loader(args, train_data, args.batch_size)
    dev_loader = get_eval_dataset_loader(args, dev_data, args.batch_size, True)
    test_loader = get_eval_dataset_loader(args, test_data, args.batch_size, True)

    path_to_pred_npy_path = {}
    for loader in [train_loader, dev_loader, test_loader]:
        for batch in tqdm.tqdm(loader):
            x = batch['x'].cuda()
            y = batch['y'].cpu().numpy().tolist()
            z = model.encode_input(x)[-1].mean(dim=1).cpu().numpy()

            for j in range(len(z)):
                path = batch['path'][j]
                predicted_path = get_nearest_neighbor_in_encoded_data(z[j], os.path.join(args.encoded_data_dir, 'chexpert'), reduce_mean=True)
                path_to_pred_npy_path[path]= predicted_path

    ## Save real paths, args, and model for future eval
    if not os.path.exists(args.submission_dir):
        os.mkdir(args.submission_dir)
    json.dump(path_to_pred_npy_path, open(os.path.join(args.submission_dir, 'parallel_data_predictions.json' ), 'w'))


def real_world_challenge(args, augmentations, test_augmentations, model):
    if os.path.exists( os.path.join(args.submission_dir, 'out_of_domain_npy_path_to_orig_path_dict.json' )):
        print("Real world (no-parallel data) submission already created. Skipping creation..")
        return

    print("### Prepare real world (no-parallel data) Challenge submission")
    dataset_name = 'mimic_cxr_edema'

    _, _, test_data = dataset_factory.get_dataset_by_name(dataset_name, args, augmentations, test_augmentations)
    loader = get_eval_dataset_loader(args, test_data, args.batch_size, True)

    paths = {}
    idx =  0
    for batch in tqdm.tqdm(loader):
        x = batch['x'].cuda()
        y = batch['y'].cpu().numpy().tolist()
        z = model.encode_input(x)[-1].mean(dim=1).cpu().numpy()

        if not os.path.exists(args.submission_dir):
            os.mkdir(args.submission_dir)
        if not os.path.exists(os.path.join(args.submission_dir, 'mimic')):
            os.mkdir(os.path.join(args.submission_dir, 'mimic'))

        for j in range(len(z)):
            path = batch['path'][j]
            npy_path = os.path.join(os.path.join(args.submission_dir, 'mimic'), '{}.npy'.format(idx) )
            idx += 1
            np.save(npy_path, z[j])
            paths[npy_path] = path
    ## Save real paths, args, and model for future eval
    json.dump(paths, open(os.path.join(args.submission_dir, 'out_of_domain_npy_path_to_orig_path_dict.json' ), 'w'))



if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = parsing.parse_args()
    main(args)
