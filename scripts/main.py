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


#Constants
DATE_FORMAT_STR = "%Y-%m-%d:%H-%M-%S"

def main(args):
    repo = git.Repo(search_parent_directories=True)
    commit  = repo.head.object
    args.commit = commit.hexsha
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
    train_data, dev_data, test_data = dataset_factory.get_dataset(args, augmentations, test_augmentations)

    train_loader = get_train_dataset_loader(args, train_data, args.batch_size)
    dev_loader = get_eval_dataset_loader(args, dev_data, args.batch_size, True)
    test_loader = get_eval_dataset_loader(args, test_data, args.batch_size, False)
    eval_train_loader = get_eval_dataset_loader(args, train_data, args.batch_size, False)

    model = lightning.get_lightning_model(args)

    log.info("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if attr not in ['optimizer_state']:
            log.info("\t{}={}".format(attr.upper(), value))

    save_path = args.results_path
    log.info("\n")
    if args.train:
        log.info("-------------\nTrain")
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=dev_loader)
        model_path = trainer.checkpoint_callback.best_model_path
        if args.distributed_backend != 'ddp' or trainer.global_rank == 0:
            log.info("Best model saved to : {}".format(model_path))
            model = model.load_from_checkpoint(model_path, args=args)
        args.model_path = model_path

    log.info("\n")
    if args.dev or args.store_hiddens:
        log.info("-------------\nDev")
        model.save_prefix = 'dev_'
        trainer.test(model, test_dataloaders=dev_loader)
    log.info("\n")
    if args.test or args.store_hiddens:
        log.info("-------------\nTest")
        model.save_prefix = 'test_'
        trainer.test(model, test_dataloaders=test_loader)

    if args.private and args.private_switch_encoder:
        log.info("-------------\nTransfer")
        model.save_prefix = 'transfer_test_'
        model.secure_encoder_0 = model.secure_encoder_1
        trainer.test(model, test_dataloaders=test_loader)

    if args.rlc_cxr_test:
        stems = ['mimic', 'stanford']
        exper_stem = args.dataset.split('_')[0]
        for stem in stems:
            model.save_prefix = 'test_{}'.format(stem)
            external_test = args.dataset.replace(exper_stem, stem)
            _, _, exter_test_data = dataset_factory.get_dataset_by_name(external_test, args, augmentations, test_augmentations)
            ext_test_loader = get_eval_dataset_loader(args, exter_test_data, args.batch_size, False)
            log.info("-------------\nTest {}".format(external_test))
            trainer.test(model, test_dataloaders=ext_test_loader)




    if args.store_hiddens or args.eval_train:
        log.info("---\n Now running Eval on train to store final hiddens for each train sample...")
        model.save_prefix = 'eval_train_'
        trainer.test(model, test_dataloaders=eval_train_loader)
    log.info("Saving args to {}".format(args.results_path))
    pickle.dump(vars(args), open(args.results_path,'wb'))

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = parsing.parse_args()
    main(args)
