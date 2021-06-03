from sandstone.learn.metrics.factory import RegisterMetric
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sandstone.learn.metrics.stats import confidence_interval
import numpy as np
import pdb

EPSILON = 1e-6
BINARY_CLASSIF_THRESHOLD = 0.5
CONFIDENCE_INTERVAL = 0.95
NUM_BOOTSTRAP = 5000


@RegisterMetric("classification")
def get_accuracy_metrics(logging_dict, args):
    stats_dict = OrderedDict()

    probs = np.array(logging_dict['probs'])
    preds = probs.argmax(axis=-1).reshape(-1)
    golds = np.array(logging_dict['golds']).reshape(-1)
    probs = probs.reshape( (-1, probs.shape[-1]))
    stats_dict['accuracy'] = accuracy_score(y_true=golds, y_pred=preds)

    if args.num_classes == 2:
        stats_dict['precision'] = precision_score(y_true=golds, y_pred=preds)
        stats_dict['recall'] = recall_score(y_true=golds, y_pred=preds)
        stats_dict['f1'] = f1_score(y_true=golds, y_pred=preds)
        num_pos = golds.sum()
        if num_pos > 0 and num_pos < len(golds) and args.num_classes == 2:
            stats_dict['auc'] = roc_auc_score(golds, probs[:,-1], average='samples')
            emperical_dist = list(zip(golds, probs[:,-1]))
            def auc_estimator(emperical_dist):
                golds, probs = [e[0] for e in emperical_dist], [e[1] for e in emperical_dist]
                return roc_auc_score(golds, probs, average='samples')
            auc_low, auc_high = confidence_interval(CONFIDENCE_INTERVAL, NUM_BOOTSTRAP, emperical_dist, auc_estimator)
            stats_dict['auc_95_CI_low'], stats_dict['auc_95_CI_high'] = auc_low, auc_high

    if args.num_classes >100:
        sorted_pred = np.argsort(probs, axis=1, kind='mergesort')[:, ::-1]
        for k in [5,10,20,50]:
            top_k_score = (golds == sorted_pred[:, :k].T).any(axis=0).mean()
            stats_dict['top_{}_accuracy'.format(k)] = top_k_score

    return stats_dict