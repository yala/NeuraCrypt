import argparse
import torch
import os
import pwd
from sandstone.datasets.factory import get_dataset_class
from pytorch_lightning import Trainer

EMPTY_NAME_ERR = 'Name of augmentation or one of its arguments cant be empty\n\
                  Use "name/arg1=value/arg2=value" format'
BATCH_SIZE_SPLIT_ERR = 'batch_size (={}) should be a multiple of batch_splits (={})'
INVALID_IMG_TRANSFORMER_SPEC_ERR = 'Invalid image transformer embedding args. Must be length 3, as [name/size=value/dim=value]. Received {}'
INVALID_IMG_TRANSFORMER_EMBED_SIZE_ERR = 'Image transformer embeddings have different embedding dimensions {}'
POSS_VAL_NOT_LIST = 'Flag {} has an invalid list of values: {}. Length of list must be >=1'
INVALID_DATASET_FOR_SURVIVAL = "A dataset with '_full_future'  can only be used with survival_analysis_setup and viceversa."
NPZ_MULTI_IMG_ERROR = "Npz loading code assumes multi images are in one npz and code is only in multi-img code flow."
SELF_SUPER_ERROR = "Moco and Byol only supported with instance disrimination task. Must be multi image with 2 images"

def parse_augmentations(raw_augmentations):
    """
    Parse the list of augmentations, given by configuration, into a list of
    tuple of the augmentations name and a dictionary containing additional args.

    The augmentation is assumed to be of the form 'name/arg1=value/arg2=value'

    :raw_augmentations: list of strings [unparsed augmentations]
    :returns: list of parsed augmentations [list of (name,additional_args)]

    """
    augmentations = []
    for t in raw_augmentations:
        arguments = t.split('/')
        name = arguments[0]
        if name == '':
            raise Exception(EMPTY_NAME_ERR)

        kwargs = {}
        if len(arguments) > 1:
            for a in arguments[1:]:
                splited = a.split('=')
                var = splited[0]
                val = splited[1] if len(splited) > 1 else None
                if var == '':
                    raise Exception(EMPTY_NAME_ERR)

                kwargs[var] = val

        augmentations.append((name, kwargs))

    return augmentations

def parse_embeddings(raw_embeddings):
    """
    Parse the list of embeddings, given by configuration, into a list of
    tuple of the embedding embedding_name, size ('vocab size'), and the embedding dimension.

    :raw_embeddings: list of strings [unparsed transformers], each of the form 'embedding_name/size=value/dim=value'
    :returns: list of parsed embedding objects [(embedding_name, size, dim)]

    For example:
        --hidden_transformer_embeddings time_seq/size=10/dim=32 view_seq/size=2/dim=32 side_seq/size=2/dim=32
    returns
        [('time_seq', 10, 32), ('view_seq', 2, 32), ('side_seq', 2, 32)]
    """
    embeddings = []
    for t in raw_embeddings:
        arguments = t.split('/')
        if len(arguments) != 3:
                raise Exception(INVALID_IMG_TRANSFORMER_SPEC_ERR.format(len(arguments)))
        name = arguments[0]
        size = arguments[1].split('=')[-1]
        dim = arguments[2].split('=')[-1]

        embeddings.append((name, int(size), int(dim)))

    if not all([embed[-1] == int(dim) for embed in embeddings]):
        raise Exception(INVALID_IMG_TRANSFORMER_EMBED_SIZE_ERR.format([embed[-1] for embed in embeddings]))
    return embeddings



def parse_dispatcher_config(config):
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
    but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of flag strings, each of which encapsulates one job.
        *Example: --train --cuda --dropout=0.1 ...
    returns: experiment_axies - axies that the grid search is searching over
    '''
    jobs = [""]
    experiment_axies = []
    search_spaces = config['search_space']

    # Support a list of search spaces, convert to length one list for backward compatiblity
    if not isinstance(search_spaces, list):
        search_spaces = [search_spaces]


    for search_space in search_spaces:
        # Go through the tree of possible jobs and enumerate into a list of jobs
        for ind, flag in enumerate(search_space):
            possible_values = search_space[flag]
            if len(possible_values) > 1:
                experiment_axies.append(flag)

            children = []
            if len(possible_values) == 0 or type(possible_values) is not list:
                raise Exception(POSS_VAL_NOT_LIST.format(flag, possible_values))
            for value in possible_values:
                for parent_job in jobs:
                    if type(value) is bool:
                        if value:
                            new_job_str = "{} --{}".format(parent_job, flag)
                        else:
                            new_job_str = parent_job
                    elif type(value) is list:
                        val_list_str = " ".join([str(v) for v in value])
                        new_job_str = "{} --{} {}".format(parent_job, flag,
                                                          val_list_str)
                    else:
                        new_job_str = "{} --{} {}".format(parent_job, flag, value)
                    children.append(new_job_str)
            jobs = children

    return jobs, experiment_axies

def parse_args(args_strings=None):
    parser = argparse.ArgumentParser(description='Sandstone research repo.')
    # setup
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    parser.add_argument('--dev', action='store_true', default=False, help='Whether or not to run model on dev set')
    parser.add_argument('--eval_train', action='store_true', default=False, help='Whether or not to evaluate model on train set')
    parser.add_argument('--fine_tune', action='store_true', default=False, help='Whether or not to fine_tune model')
    parser.add_argument('--num_epochs_fine_tune', type=int, default=1, help='Num epochs to finetune model')
    parser.add_argument('--lightning_name', type=str, default='default', help="Name of lightning module to structure training.")
    parser.add_argument('--debug', action='store_true', default=False, help='Set sandstone to debug mode. Load only 1000 rows in metadata, set num workers to 0, max train and dev small.')
    parser.add_argument('--num_steps_alt_optimization', type=int, default=0, help='Number of steps to train alt model per training main model.')

    # data
    parser.add_argument('--dataset', default='mnist', help='Name of dataset from dataset factory to use [default: mnist]')
    parser.add_argument('--ref_dataset', default='stanford_cxr_edema', help='Name of dataset from dataset factory to use [default: mnist]')
    parser.add_argument('--private', action='store_true', default=False, help='Whether to use secure encoding scheme')
    parser.add_argument('--private_kernel_size', type=int, default=16, help='Kernel size for private conv layer. Can also be thought of as the patch size as first conv is non-overlapping')
    parser.add_argument('--private_depth', type=int, default=4, help='Depth to NeuraCrypt encoder. We note that since the first conv, positional embedding and last convs are always sampled, the depth of the NeuralCrypt encoder is (private_depth + 3). Set this flag to -1 to use a single linear conv as the encoder (i.e a weak encoder) ')
    parser.add_argument('--private_switch_encoder', action='store_true', default=False, help='Switch private encoders for a final test. Used for transfer learning attack')
    parser.add_argument('--rlc_cxr_test', action='store_true', default=False, help='If true, also test on all hosptial versions of this dataset obj')
    parser.add_argument('--rlc_private_multi_host', action='store_true', default=False, help='If true, use diff key for all hospitals, i.e, private collaborative training')
    parser.add_argument('--encoded_data_dir', type=str, default='/Mounts/rbg-storage1/users/adamyala/neuracrypt_embeddings/sandbox', help='dir to store encoded images for export.')

    parser.add_argument('--image_augmentations', nargs='*', default=['scale_2d'], help='List of image-transformations to use [default: ["scale_2d"]] \
                        Usage: "--image_augmentations trans1/arg1=5/arg2=2 trans2 trans3/arg4=val"')
    parser.add_argument('--tensor_augmentations', nargs='*', default=['normalize_2d'], help='List of tensor-transformations to use [default: ["normalize_2d"]]\
                        Usage: similar to image_augmentations')
    parser.add_argument('--test_image_augmentations', nargs='*', default=['scale_2d'], help='List of image-transformations to use for the dev and test dataset [default: ["scale_2d"]] \
                        Usage: similar to image_augmentations')
    parser.add_argument('--test_tensor_augmentations', nargs='*', default=['normalize_2d'], help='List of tensor-transformations to use for the dev and test dataset [default: ["normalize_2d"]]\
                        Usage: similar to image_augmentations')
    parser.add_argument('--fix_seed_for_multi_image_augmentations', action='store_true', default=False, help='Whether to use the same seed (same random augmentations) for multi image inputs.')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for each data loader [default: 4]')
    parser.add_argument('--img_size',  type=int, nargs='+', default=[256, 256], help='width and height of image in pixels. [default: [256,256]')
    parser.add_argument('--get_dataset_stats', action='store_true', default=False, help='Whether to compute the mean and std of the training images on the fly rather than using precomputed values')
    parser.add_argument('--img_mean', type=float, nargs='+', default=[0.2023], help='mean value of img pixels. Per channel. ')

    parser.add_argument('--img_std', type=float, nargs='+', default=[0.2576], help='std of img pixels. Per channel. ')
    parser.add_argument('--img_dir', type=str, default='/home/administrator/Mounts/Isilon/pngs16', help='dir of images. Note, image path in dataset jsons should stem from here')
    parser.add_argument('--num_chan', type=int, default=3, help='Number of channels in img. [default:3]')
    parser.add_argument('--force_input_dim', action='store_true', default=False, help='trunctate hiddens from file if not == to input_dim')
    parser.add_argument('--input_dim', type=int, default=512, help='Input dim for 2stage models. [default:512]')
    parser.add_argument('--multi_image', action='store_true', default=False, help='Whether image will contain multiple slices. Slices could indicate different times, depths, or views')
    parser.add_argument('--input_loader_name', type=str, default=None, help = "Name of loader to use (images, hiddens, etc)")
    parser.add_argument('--load_img_as_npz', action='store_true', default=False, help='All channels of images are stored as one npz')
    parser.add_argument('--load_IRS_as_npy', action='store_true', default=False, help='Load thermal video IRS as NPY')
    parser.add_argument('--use_random_offset', action='store_true', default=False, help='Load thermal video IRS as NPY')
    parser.add_argument('--concat_img_channels', action='store_true', default=False, help='Whether combine images across channels')
    parser.add_argument('--num_images', type=int, default=1, help='In multi image setting, the number of images per single sample.')
    parser.add_argument('--min_num_images', type=int, default=0, help='In multi image setting, the min number of images per single sample.')
    parser.add_argument('--inflation_factor', type=int, default=1, help='In multi image setting, dim of depth to inflate the model to.')
    parser.add_argument('--inflate_time_like_hw', action='store_true', default=False, help='Inflate time depths and strides like 2d')


    parser.add_argument('--metadata_dir', type=str, default='', help='dir of metadata jsons.')
    parser.add_argument('--cache_path', type=str, default=None, help='dir to cache images.')
    parser.add_argument('--cache_full_img', action='store_true', default=False, help='Cache full image locally as well as cachable transforms')


    # sampling
    parser.add_argument('--class_bal', action='store_true', default=False, help='Wether to apply a weighted sampler to balance between the classes on each batch.')
    # regularization
    parser.add_argument('--use_adv', action='store_true', default=False, help='Wether to add a adversarial loss representing the kl divergernce from source to target domain.')
    parser.add_argument('--use_mmd_adv', action='store_true', default=False, help='Wether to add a adversarial loss representing the mmd distance from source to target domain. only used if --use_adv is set to true to override KL discrim')
    parser.add_argument('--use_plaintext_attack', action='store_true', default=False, help='Use standard plaintext attack.')
    parser.add_argument('--remove_pixel_shuffle', action='store_true', default=False, help='Remove pixel shuffle for mmd.')
    parser.add_argument('--use_weak_encoder', action='store_true', default=False, help='Set true to use simple linear model as priv encoder.')
    parser.add_argument('--use_same_dist', action='store_true', default=False, help='Use same dist of samples for both private and public.')
    parser.add_argument('--use_shuffle_pairs', action='store_true', default=False, help='Use fixed shuffle list mapping from private to public.')
    parser.add_argument('--attack_from_noise', action='store_true', default=False, help='Use guass noise instead of public images as source of attack.')
    parser.add_argument('--attack_snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')

    parser.add_argument('--primary_loss_lambda', type=float, default=1.0,  help='lambda to weigh the primary loss.')
    # storing hiddens
    parser.add_argument('--store_hiddens', action='store_true', default=False, help='Save hidden repr from each image to an npz based off results path, git hash and exam name')
    parser.add_argument('--save_predictions', action='store_true', default=False, help='Save hidden repr from each image to an npz based off results path, git hash and exam name')
    parser.add_argument('--hiddens_dir', type=str, default='hiddens/test_run', help='Dir to store hiddens npy"s when store_hiddens is true')

    # learning
    parser.add_argument('--optimizer', type=str, default="adam", help='optimizer to use [default: adam]')
    parser.add_argument('--objective', type=str, default="cross_entropy", help='objective function to use [default: cross_entropy]')
    parser.add_argument('--init_lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum to use with SGD')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='initial learning rate [default: 0.5]')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 Regularization penaty [default: 0]')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs without improvement on dev before halving learning rate and reloading best model [default: 5]')

    parser.add_argument('--tuning_metric', type=str, default='loss', help='Metric to judge dev set results. Possible options include auc, loss, accuracy [default: loss]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 128]')
    parser.add_argument('--dropout', type=float, default=0.25, help='Amount of dropout to apply on last hidden layer [default: 0.25]')
    parser.add_argument('--save_dir', type=str, default='snapshot', help='where to dump the model')
    parser.add_argument('--results_path', type=str, default='logs/test.args', help='where to save the result logs')
    parser.add_argument('--project_name', type=str, default='sandstone-sandbox', help='Name of project for comet logger')
    parser.add_argument('--workspace', type=str, default='username', help='Name of workspace for comet logger')
    parser.add_argument('--comet_tags', nargs='*', default=[], help="List of tags for comet logger")
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of data to use, i.e 1.0 for all and 0 for none. Used for learning curve analysis.')

    # Alternative training/testing schemes
    parser.add_argument('--cross_val_seed', type=int, default=0, help="Seed used to generate the partition.")
    parser.add_argument('--model_name', type=str, default='resnet18', help="Form of model, i.e resnet18, aggregator, revnet, etc.")
    parser.add_argument('--num_layers', type=int, default=3, help="Num layers for transformer based models.")
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    parser.add_argument('--state_dict_path', type=str, default=None, help='filename of model snapshot to load[default: None]')

    # transformer
    parser.add_argument('--hidden_dim', type=int, default=512, help='start hidden dim for transformer')
    parser.add_argument('--num_heads', type=int, default=8, help='Num heads for transformer')
    # resnet-specific
    parser.add_argument('--block_widening_factor', type=int, default=1, help='Factor by which to widen hidden dim.')
    parser.add_argument('--pool_name', type=str, default='GlobalAvgPool', help='Pooling mechanism')


    # run
    parser = Trainer.add_argparse_args(parser)
    if args_strings is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_strings)
    args.lr = args.init_lr

    if (isinstance(args.gpus, str) and len(args.gpus.split(",")) > 1) or (isinstance(args.gpus, int) and  args.gpus > 1):
        args.distributed_backend = 'ddp'
        args.replace_sampler_ddp = False

    if args.debug:
        args.num_workers = 0
        args.limit_train_batches = 10
        args.limit_val_batches = 10
        args.limit_test_batches = 0.1


       # Set args particular to dataset
    get_dataset_class(args.dataset).set_args(args)

    args.unix_username = pwd.getpwuid( os.getuid() )[0]

    if args.private:
        args.lightning_name = 'private'

    if args.use_adv:
        args.lightning_name = 'adversarial_attack'
        args.tuning_metric = None
    if args.private_depth < 0:
        args.use_weak_encoder = True

    # learning initial state
    args.step_indx = 1

    # Parse list args to appropriate data format
    parse_list_args(args)


    return args


def parse_list_args(args):
    """Converts list args to their appropriate data format.

    Includes parsing image dimension args, augmentation args,
    block layout args, and more.

    Arguments:
        args(Namespace): Config.

    Returns:
        args but with certain elements modified to be in the
        appropriate data format.
    """

    args.image_augmentations = parse_augmentations(args.image_augmentations)
    args.tensor_augmentations = parse_augmentations(args.tensor_augmentations)
    args.test_image_augmentations = parse_augmentations(args.test_image_augmentations)
    args.test_tensor_augmentations = parse_augmentations(args.test_tensor_augmentations)

