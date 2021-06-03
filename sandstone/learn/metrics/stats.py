import numpy as np
import tqdm
import sklearn.metrics
import warnings
import pdb

RESAMPLE_FAILED_WARNING = "Resampling distrubution for estimator {} failed because of : {}"
CONFIDENCE_INTERVAL_EXCEPTION = "Cannot calculate confidence interval. Sampled {} times for found {} (target {}) valid samples."

def ci_bounds(mean, deltas, confidence_interval, num_resamples):
    '''
    Returns the lower and upper bounds of the confidence interval.
    mean : the empirical mean
    deltas : deltas from the empirical mean from different samples
    '''
    deltas = np.sort(deltas)
    index_offset = int((1 - confidence_interval)/2. * num_resamples)
    lower_delta, upper_delta = deltas[index_offset], deltas[-index_offset]
    lower_bound, upper_bound = mean - upper_delta, mean - lower_delta
    return lower_bound, upper_bound


def confidence_interval(confidence_interval, num_resamples, emperical_distribution, estimator=np.mean, clusters=None):
    '''
    Estimates confidence interval of the mean of emperical_distribution
    using emperical bootstrap. Method details are available at:

    https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf


    -confidence_interval: Amount of probability mass interval should cover.
    -num_resamples: Amount of trails to use to estimate confidence interval.
    Should as large as computationally feasiable.
    -emperical_distribution: Emperical distributions for which we want to
    estimate the confidence interval of the mean

    if Clusters is not none, perform two stage bootstrap.
    '''

    emp_mean = estimator(emperical_distribution)
    num_samples = len(emperical_distribution)
    deltas = []
    num_tries = 0
    if clusters is not None:
        cluster_to_inds = {}
        for ind, cluster in enumerate(clusters):
            if cluster not in cluster_to_inds:
                cluster_to_inds[cluster] = []
            cluster_to_inds[cluster].append(ind)
        num_clusters = len(cluster_to_inds)
        cluster_ids = list(cluster_to_inds.keys())

    while len(deltas) < num_resamples:
        if num_tries > num_resamples * 10:
            raise Exception(CONFIDENCE_INTERVAL_EXCEPTION.format(num_tries, len(deltas), num_resamples))
        num_tries += 1
        try:
            if clusters is None:
                resample_ind = np.random.choice(a=range(len(emperical_distribution)),
                                        size=num_samples,
                                        replace=True
                                        )
            else:
                resample_clusters = np.random.choice(a=cluster_ids,
                                                    size=num_clusters,
                                                    replace=True)
                resample_ind = []
                for cluster in resample_clusters:
                    resampled_inds_for_cluster = np.random.choice(a=cluster_to_inds[cluster],
                                                                size=len(cluster_to_inds[cluster]),
                                                                replace=True)
                    resample_ind.extend(resampled_inds_for_cluster)

            resample_dist = [ emperical_distribution[ind] for ind in resample_ind]
            delta = estimator(resample_dist) - emp_mean
            deltas.append( delta )
        except Exception as e:
            warnings.warn(RESAMPLE_FAILED_WARNING.format(estimator, e))

    lower_bound, upper_bound = ci_bounds(emp_mean, deltas, confidence_interval, num_resamples)

    return (lower_bound, upper_bound)

def resample_set_by_distribution(sets, probs, length):
    '''
    Returns a new set by sampling from the given sets with repetitions by the given probs.
    sets : list of sets to sample from
    probs : list of probability for each set (should sum to 1)
    length : length of the new built set
    Returns the new set and an np.array with the set index for each element in the new array.
    '''

    probs_array = np.array([])
    concat_array = np.array([])
    set_inds = np.array([])
    for i, cur_set in enumerate(sets):
        cur_probs = np.ones((len(cur_set))) * probs[i] / len(cur_set)
        probs_array = np.append(probs_array, cur_probs)
        concat_array = np.append(concat_array, cur_set)
        set_inds = np.append(set_inds, np.ones(len(cur_set))*i)

    idx = np.random.choice(np.arange(len(concat_array)), size=length, replace=True, p=probs_array)
    return concat_array[idx], set_inds[idx]
