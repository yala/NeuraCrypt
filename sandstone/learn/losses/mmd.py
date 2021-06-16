import torch
import torch.nn as nn
from sandstone.learn.losses.factory import RegisterLoss
from collections import OrderedDict
import pdb

SIGMAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
            1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6]




def compute_pairwise_distances(x, y):
    """ Computes the squared pairwise Euclidean distances between x and y.
    Args:
      x: a tensor of shape [num_x_samples, num_features]
      y: a tensor of shape [num_y_samples, num_features]
    Returns:
      a distance matrix of dimensions [num_x_samples, num_y_samples]
    Raise:
      ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.size()) == len(y.size()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.size()[1] != y.size()[1]:
        raise ValueError('The number of features should be the same.')

    # By making the `inner' dimensions of the two matrices equal to 1 using
    # broadcasting then we are essentially substracting every pair of rows
    # of x and y.
    norm = lambda x: torch.sum(x * x, 1)
    return norm(x.unsqueeze(2) - y.t())

def gaussian_kernel(x, y, sigmas):
    """ Computes a Gaussian RBK between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      sigmas: a tensor of floats which denote the widths of each of the
        gaussians in the kernel.
    Returns:
      A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel
    """
    beta = 1. / (2. * (sigmas.unsqueeze(1)))

    dist = compute_pairwise_distances(x, y)

    s = torch.matmul(beta, dist.view(1, -1))
    return (torch.sum(torch.exp(-s), 0)).view_as(dist)


@RegisterLoss("mmd_loss")
def get_mmd_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()

    sigmas = torch.nn.Parameter(torch.FloatTensor(SIGMAS), requires_grad=False).to(batch['x'].device)
    x = batch['x'].reshape( [batch['x'].size()[0], -1]) if args.remove_pixel_shuffle else batch['x'].mean(dim=1)

    B, C = x.size()
    if args.load_data_from_encoded_dir:
        gen = x
        real = batch['z'].mean(dim=1) if not args.remove_pixel_shuffle else batch['z'].reshape( [batch['z'].size()[0], -1])
    else:
        is_real = batch['source'].bool().unsqueeze(-1)
        is_gen = ~ is_real
        real = torch.masked_select(x, is_real).view([-1, C])
        gen = torch.masked_select(x, is_gen).view([-1, C])

    cost = torch.mean(gaussian_kernel(real, real, sigmas))
    cost += torch.mean(gaussian_kernel(gen, gen, sigmas))
    cost -= 2 * torch.mean(gaussian_kernel(real, gen, sigmas))

    cost = torch.clamp(cost, min=0)

    logging_dict['mmd_loss'] = cost.detach()
    return cost * args.primary_loss_lambda, logging_dict, predictions

