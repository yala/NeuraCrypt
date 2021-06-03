import pytorch_lightning as pl
import torch
from pytorch_lightning import _logger as log
import sandstone.models.factory as model_factory
import sandstone.learn.metrics.factory as metric_factory
from sandstone.utils.generic import get_path_for_x, concat_all_gather
from sandstone.learn.losses.factory import get_loss
from sandstone.learn.lightning.factory import RegisterLightning
from collections import OrderedDict
from argparse import Namespace
import pickle
from sandstone.models.pools.factory import get_pool
import copy
import remote_pdb

@RegisterLightning("default")
class Sandstone(pl.LightningModule):
    '''
    Lightning Module
    Methods:
        .log/.log_dict: log inputs to logger
    Notes:
        *_epoch_end method returns None
        self can log additional data structures to logger with self.logger.experiment.log_* (*= 'text', 'image', 'audio', 'confusion_matrix', 'histogram')
    '''
    def __init__(self, args):
        super(Sandstone, self).__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
        self.args = args
        self.model = model_factory.get_model_by_name(args.model_name, args)
        self.save_prefix = 'default'
        self.save_hyperparameters()

    def set_finetune(self, finetune_flag):
        return

    def forward(self, x, batch):
        return self.model(x, batch)

    def step(self, batch, batch_idx, optimizer_idx, log_key_prefix = ""):
        model_output = self.model(batch['x'], batch=batch)
        logging_dict, predictions_dict = OrderedDict(), OrderedDict()

        if 'exam' in batch:
            predictions_dict['exam'] = batch['exam']
        if 'y' in batch:
            predictions_dict['golds'] = batch['y']

        loss_fns = self.get_loss_functions(self.args)
        loss = 0
        for loss_fn_name in loss_fns:
            local_loss, local_log_dict, local_predictions_dict = get_loss(loss_fn_name)(model_output, batch, self, self.args)
            loss += local_loss
            logging_dict.update(local_log_dict)
            predictions_dict.update(local_predictions_dict)
        logging_dict = prefix_dict(logging_dict, log_key_prefix)
        predictions_dict = prefix_dict(predictions_dict, log_key_prefix)
        return loss, logging_dict, predictions_dict, model_output

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        result = OrderedDict()
        loss, logging_dict, predictions_dict, _ = self.step(batch, batch_idx, optimizer_idx, log_key_prefix="train_")
        logging_dict['train_loss'] = loss.detach()
        self.log_dict(logging_dict, prog_bar = False, on_step=True, on_epoch=True)
        result['logs'] = logging_dict
        self.log_tensor_dict(predictions_dict, prog_bar = False, logger=False)
        result.update(predictions_dict)
        # lightning expects 'loss' key in output dict. ow loss := None by default
        result['loss'] = loss
        return result

    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        result = OrderedDict()
        loss, logging_dict, predictions_dict, _= self.step(batch, batch_idx, optimizer_idx, log_key_prefix="val_")
        logging_dict['val_loss'] = loss.detach()
        self.log_dict(logging_dict, prog_bar =True, sync_dist=True)
        result['logs'] = logging_dict
        if self.args.distributed_backend == 'ddp':
            predictions_dict = gather_predictions_dict(predictions_dict)
        self.log_tensor_dict(predictions_dict, prog_bar = False, logger=False)
        result.update(predictions_dict)
        return result

    def test_step(self, batch, batch_idx, optimizer_idx=None):
        result = OrderedDict()
        loss, logging_dict, predictions_dict, model_output = self.step(batch, batch_idx, optimizer_idx, log_key_prefix=self.save_prefix)
        logging_dict['{}_loss'.format(self.save_prefix)] = loss.detach()
        result['logs'] = logging_dict
        self.log_tensor_dict(predictions_dict, prog_bar = False, logger=False)
        if self.args.store_hiddens:
            if not 'exams' in predictions_dict:
                if self.global_rank == 0 and batch_idx == 0:
                    log.warn("Warning. Cannot store hiddens without exams being defined. Skipping store_hiddens logic")
                return result
            for exam, hidden in zip(predictions_dict['exams'], model_output['hidden']):
                exam_filename = get_path_for_x(exam, self.args)
                self.write(hidden, exam_filename)
        result.update(predictions_dict)
        return result

    def training_epoch_end(self, outputs):
        if len(outputs) == 0:
            return
        outputs = gather_step_outputs(outputs)
        # loss already logged in progress_bar_dict (get_progress_bar_dict()), and logging twice creates issue
        del outputs['loss']
        epoch_metrics = metric_factory.compute_epoch_metrics( outputs, self.args, self.device, key_prefix="train_")
        for k,v in outputs['logs'].items():
            epoch_metrics[k] = v.mean()
        self.log_dict(epoch_metrics, prog_bar = True, logger=True)

    def validation_epoch_end(self, outputs):
        if len(outputs) == 0:
            return
        outputs = gather_step_outputs(outputs)
        epoch_metrics = metric_factory.compute_epoch_metrics( outputs, self.args, self.device, key_prefix="val_")
        for k,v in outputs['logs'].items():
            epoch_metrics[k] = v.mean()
        self.log_dict(epoch_metrics, prog_bar = True, logger=True)

    def test_epoch_end(self, outputs):
        if len(outputs) == 0:
            return
        outputs = gather_step_outputs(outputs)
        epoch_metrics = metric_factory.compute_epoch_metrics( outputs, self.args, self.device, key_prefix=self.save_prefix)
        for k,v in outputs['logs'].items():
            epoch_metrics[k] = v.mean()

        self.log_dict(epoch_metrics, prog_bar = True, logger=True)

        ## Dump metrics for use by dispatcher
        metrics = {k[len(self.save_prefix):] :v.mean().item() for k,v in outputs.items() if 'loss' in k}
        metrics.update({k[len(self.save_prefix):] :v.mean().item() for k,v in epoch_metrics.items()})
        metrics_filename = "{}.{}.metrics".format(self.args.results_path, self.save_prefix)
        pickle.dump(metrics, open(metrics_filename,'wb'))
        if self.args.save_predictions:
            predictions_dict = {k:v.cpu() if isinstance(v, torch.Tensor) else v for k,v in outputs.items()}
            predictions_filename = "{}.{}.predictions".format(self.args.results_path, self.save_prefix)
            pickle.dump(predictions_dict, open(predictions_filename,'wb'))

    def configure_optimizers(self):
        return model_factory.get_optimizer(self.model, self.args)

    def log_tensor_dict(self, output,  prog_bar = False, logger=True, on_step=None, on_epoch=None, sync_dist=False):
        dict_of_tensors = {k:v.float() for k,v in output.items() if isinstance(v, torch.Tensor) }
        self.log_dict(dict_of_tensors, prog_bar = prog_bar, logger=logger, on_step=on_step, on_epoch=on_epoch, sync_dist=sync_dist)

    def get_loss_functions(self, args):
        if hasattr(self.model, "get_loss_functions"):
            return self.model.get_loss_functions(args)
        loss_fns =  ['mse'] if args.num_classes == 1 \
                    else ['cross_entropy']

        return loss_fns

def prefix_dict(d, prefix):
    r = OrderedDict()
    for k,v in d.items():
        r[prefix+k] = v
    return r

def gather_predictions_dict(predictions):
    gathered_preds = {k: concat_all_gather(v) if isinstance(v, torch.Tensor) else v for k,v in predictions.items()}
    return gathered_preds

def gather_step_outputs(outputs):
    output_dict = OrderedDict()
    if isinstance(outputs[-1], list):
        outputs = outputs[0]

    for k in outputs[-1].keys():
        if k == "logs":
            output_dict[k] = gather_step_outputs([output['logs'] for output in outputs])
        elif isinstance(outputs[-1][k], torch.Tensor) and len(outputs[-1][k].shape) == 0:
            output_dict[k] = torch.stack([output[k] for output in outputs])
        elif isinstance(outputs[-1][k], torch.Tensor):
            output_dict[k] = torch.cat([output[k] for output in outputs], dim = 0)
        else:
            output_dict[k] = [output[k] for output in outputs]
    return output_dict
