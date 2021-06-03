import torch
from torch import nn
from sandstone.utils.generic import log
import pdb
MODEL_REGISTRY = {}

STRIPPING_ERR = 'Trying to strip the model although last layer is not FC.'
NO_MODEL_ERR = 'Model {} not in MODEL_REGISTRY! Available models are {} '
NO_OPTIM_ERR = 'Optimizer {} not supported!'

def RegisterModel(model_name):
    """Registers a configuration."""

    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator


def get_model(args):
    return get_model_by_name(args.model_name, args)


def get_model_by_name(name, args):
    '''
        Get model from MODEL_REGISTRY based on args.model_name
        args:
        - name: Name of model, must exit in registry
        - allow_wrap_model: whether or not override args.wrap_model and disable model_wrapping.
        - args: run ime args from parsing

        returns:
        - model: an instance of some torch.nn.Module
    '''
    if not name in MODEL_REGISTRY:
        raise Exception(
            NO_MODEL_ERR.format(
                name, MODEL_REGISTRY.keys()))


    model = MODEL_REGISTRY[name](args)
    return wrap_model(model, args)

def wrap_model(model, args, allow_data_parallel=True):
    return model

def load_model(path, args):
    log('\nLoading model from [%s]...' % path, args)
    try:
        model = torch.load(path, map_location=args.map_location)
    except:
        raise Exception(
            "Sorry, snapshot {} does not exist!".format(path))

    if isinstance(model, dict) and 'model' in model:
        model = model['model']
    if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module.cpu()
    if hasattr(model, 'args'):
        model.args = args
    return model



def get_optimizer(model, args):
    '''
    Helper function to fetch optimizer based on args.
    '''
    params = [param for param in model.parameters() if param.requires_grad]
    if args.optimizer == 'adam':
        optimizer =  torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer =  torch.optim.SGD(params,
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum )
    else:
        raise Exception(NO_OPTIM_ERR.format(args.optimizer))
    
    if args.tuning_metric is not None:
        scheduler =  {
             'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                            patience= args.patience, factor= args.lr_decay,\
                            mode = 'min' if 'loss' in args.tuning_metric else 'max'),
             'monitor': 'val_{}'.format(args.tuning_metric),
             'interval': 'epoch',
             'frequency': 1
          }
        return [optimizer], [scheduler]
    else:
        return [optimizer], []
