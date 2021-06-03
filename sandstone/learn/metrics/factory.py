from collections import OrderedDict, defaultdict
import numpy as np
import torch
METRIC_REGISTRY = {}

NO_METRIC_ERR = 'Metric {} not in METRIC_REGISTRY! Available metrices are {}'

def RegisterMetric(metric_name):
    """Registers a metric."""

    def decorator(f):
        METRIC_REGISTRY[metric_name] = f
        return f

    return decorator

def get_metric(metric_name):
    """Get metric from METRIC_REGISTRY based on metric_name."""

    if not metric_name in METRIC_REGISTRY:
        raise Exception(NO_METRIC_ERR.format(
            metric_name, METRIC_REGISTRY.keys()))

    metric = METRIC_REGISTRY[metric_name]

    return metric

def compute_epoch_metrics(result_dict, args, device, key_prefix = ""):
    stats_dict = OrderedDict()
    ## Now call additional metric functions that are specialized
    '''
        Remove prefix from keys. For instance, convert:
        val_probs -> probs for standard handling in the metric fucntions
    '''
    result_dict_wo_key_prefix = {}

    for k,v in result_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if k == 'meta':
            continue
        if  key_prefix != "" and k.startswith(key_prefix):
                k_wo_prefix = k[len(key_prefix):]
                result_dict_wo_key_prefix[k_wo_prefix] = v
        else:
            result_dict_wo_key_prefix[k] = v


    additional_metrics = []
    if args.num_classes > 1 and 'probs' in result_dict_wo_key_prefix:
        additional_metrics = ['classification']

    for metric_name in additional_metrics:
        stats_wo_prefix = get_metric(metric_name)(result_dict_wo_key_prefix, args)
        for k,v in stats_wo_prefix.items():
            stats_dict[key_prefix + k] = torch.tensor(v, device=device)

    return stats_dict

