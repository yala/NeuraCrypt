from sandstone.learn.losses.factory import RegisterLoss
import torch
import torch.nn.functional as F
import torch.nn as nn
from sandstone.utils.generic import get_base_model_obj
from collections import OrderedDict
from sandstone.utils.generic import log
import pdb

EPSILON = 1e-6

@RegisterLoss("cross_entropy")
def get_model_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    logit = model_output['logit']
    if args.lightning_name == 'private' and 'y_onehot' in batch:
        logprobs = F.log_softmax(logit, dim=-1)
        batchloss = - torch.sum( batch['y_onehot'] * logprobs, dim=1)
        loss = batchloss.mean()
    else:
        loss = F.cross_entropy(logit, batch['y'].long())
    logging_dict['cross_entropy_loss'] = loss.detach()
    predictions['probs'] = F.softmax(logit, dim=-1).detach()
    return loss * args.primary_loss_lambda, logging_dict, predictions


@RegisterLoss("mse")
def get_mse_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    pred = model_output['logit']
    loss = F.mse_loss(pred, batch['y'].view_as(pred))
    logging_dict['mse_loss'] = loss.detach()
    predictions['pred'] = pred.detach()
    return loss * args.primary_loss_lambda, logging_dict, predictions


@RegisterLoss("survival")
def get_survival_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    predictions = OrderedDict()
    assert args.survival_analysis_setup
    logit = model_output['logit']
    y_seq, y_mask = batch['y_seq'], batch['y_mask']
    loss = F.binary_cross_entropy_with_logits(logit, y_seq.float(), weight=y_mask.float(), size_average=False)/ torch.sum(y_mask.float())
    logging_dict['survival_loss'] = loss.detach()
    predictions['probs'] = F.sigmoid(logit).detach()
    predictions['censors'] = batch['time_at_event']
    return loss * args.primary_loss_lambda, logging_dict, predictions

