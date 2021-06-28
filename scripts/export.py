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
from sandstone.utils.generic import get_train_dataset_loader, get_eval_dataset_loader
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
    args.lightning_name = 'adversarial_attack'
    result_path_stem = args.results_path.split("/")[-1].split('.')[0]
    log.info("Sandstone main running from commit: \n\n{}\n{}author: {}, date: {}".format(
        commit.hexsha, commit.message, commit.author, commit.committed_date))

    snapshot_dir = os.path.join(args.save_dir, result_path_stem)
    if not os.path.exists(snapshot_dir):
        os.mkdir(snapshot_dir)
    print("snapshot_dir: {}".format(snapshot_dir))
    if args.tuning_metric is not None:
        checkpoint_callback = ModelCheckpoint(
            filepath=snapshot_dir,
            save_top_k=1,
            verbose=True,
            monitor='val_{}'.format(args.tuning_metric),
            mode='min' if 'loss' in args.tuning_metric else 'max',
            prefix=""
        )
        args.callbacks = [checkpoint_callback]
    trainer = pl.Trainer.from_argparse_args(args)
    args.num_nodes = trainer.num_nodes
    args.num_processes = trainer.num_processes
    args.world_size = args.num_nodes * args.num_processes
    args.global_rank = trainer.global_rank
    args.local_rank = trainer.local_rank

    tb_logger = pl.loggers.CometLogger()
    trainer.logger = tb_logger


    if args.get_dataset_stats:
        log.info("\nComputing image mean and std...")
        args.img_mean, args.img_std = get_dataset_stats(args)
        log.info('Mean: {}'.format(args.img_mean))
        log.info('Std: {}'.format(args.img_std))

    log.info("\nLoading data-augmentation scheme...")
    augmentations = augmentation_factory.get_augmentations(
        args.image_augmentations, args.tensor_augmentations, args)
    test_augmentations = augmentation_factory.get_augmentations(
        args.test_image_augmentations, args.test_tensor_augmentations, args)
    # Load dataset and add dataset specific information to args
    log.info("\nLoading data...")
    public_data, _, private_encoded_data = dataset_factory.get_dataset(args, augmentations, test_augmentations)

    np.random.shuffle( public_data.dataset)
    np.random.shuffle( private_encoded_data.dataset)


    public_loader = get_train_dataset_loader(args, public_data, args.batch_size)
    private_loader = get_train_dataset_loader(args, private_encoded_data, args.batch_size)


    model = lightning.get_lightning_model(args)
    log.info("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if attr not in ['optimizer_state']:
            log.info("\t{}={}".format(attr.upper(), value))

    save_path = args.results_path
    
    print("Model")
    print(model.attack_encoder)

    print("Len Target Dataset: {} \n Len Encoded Dataset: {}".format(len(public_data), len(private_encoded_data)))

    model = model.cuda()
    paths = {}
    for name, loader in [('challenge_2_target_dataset', public_loader), ('private_encoded_dataset', private_loader)]:
        idx = 0
        local_paths = []
        for batch in tqdm.tqdm(loader):
            x = batch['x'].cuda()
            y = batch['y'].cpu().numpy().tolist()
            z = model.encode_input(x)[-1].cpu().numpy()

            if not os.path.exists(args.encoded_data_dir):
                os.mkdir(args.encoded_data_dir)
            if not os.path.exists(os.path.join(args.encoded_data_dir, name)):
                os.mkdir(os.path.join(args.encoded_data_dir, name))

            for j in range(len(z)):
                path = batch['path'][j]

                npy_dir = os.path.join(args.encoded_data_dir, name)
                npy_path = os.path.join(npy_dir, '{}.npy'.format(idx) )
                np.save(npy_path, z[j])
                idx += 1
                paths[path] = npy_path
                local_paths.append(path)

        np.random.shuffle(local_paths)
        json.dump(local_paths, open(os.path.join(args.encoded_data_dir, '{}_path_list.json'.format(name) ), 'w'))

    ## Save real paths, args, and model for future eval
    json.dump(paths, open(os.path.join(args.encoded_data_dir, 'path_dict.json' ), 'w'))

    args.model_path = os.path.join(args.encoded_data_dir, 'model.p')
    pickle.dump(vars(args), open(os.path.join(args.encoded_data_dir, 'args.p' ), 'wb'))
    torch.save(model, args.model_path)




if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = parsing.parse_args()
    main(args)
