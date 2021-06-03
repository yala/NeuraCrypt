from argparse import Namespace
import pickle

LIGHTNING_REGISTRY = {}

NO_LIGHTNING_ERR = 'Lightning {} not in LIGHTNING_REGISTRY! Available lightning modules are {}'

def RegisterLightning(lightning_name):
    """Registers a lightning."""

    def decorator(f):
        LIGHTNING_REGISTRY[lightning_name] = f
        return f

    return decorator

def get_lightning(lightning_name):
    """Get lightning from LIGHTNING_REGISTRY based on lightning_name."""

    if not lightning_name in LIGHTNING_REGISTRY:
        raise Exception(NO_LIGHTNING_ERR.format(
            lightning_name, LIGHTNING_REGISTRY.keys()))

    lightning = LIGHTNING_REGISTRY[lightning_name]

    return lightning

def get_lightning_model(args):
    if args.snapshot is None:
        lightning_class = get_lightning(args.lightning_name)
        return lightning_class(args)
    else:
        snapshot_args = Namespace(**pickle.load(open(args.snapshot,'rb')))
        lightning_class = get_lightning(snapshot_args.lightning_name)
        model_w_hparams = lightning_class(snapshot_args)
        model = model_w_hparams.load_from_checkpoint(snapshot_args.model_path, args=snapshot_args)
        return model
