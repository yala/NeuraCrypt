import pickle
import pdb
NO_DATASET_ERR = "Dataset {} not in DATASET_REGISTRY! Available datasets are {}"
DATASET_REGISTRY = {}


def RegisterDataset(dataset_name):
    """Registers a dataset."""

    def decorator(f):
        DATASET_REGISTRY[dataset_name] = f
        return f

    return decorator


def get_dataset_class(name):
    if name not in DATASET_REGISTRY:
        raise Exception(
            NO_DATASET_ERR.format(name, DATASET_REGISTRY.keys()))

    return DATASET_REGISTRY[name]

def get_dataset_by_name(name, args, augmentations, test_augmentations):
    dataset_class = get_dataset_class(name)

    train = dataset_class(args, augmentations, 'train')
    dev_augmentations =  test_augmentations
    dev = dataset_class(args, dev_augmentations, 'dev')
    test = dataset_class(args, test_augmentations, 'test')

    return train, dev, test

# Depending on arg, build dataset
def get_dataset(args, augmentations, test_augmentations):
    return get_dataset_by_name(args.dataset, args, augmentations, test_augmentations)
