# This code was mostly ruthlessly appropriated from tyneumann's "Minimal PyTorch implementation of Generative Latent Optimization" https://github.com/tneumann/minimal_glo. Thank the lord for clever germans.

import numpy as np

import torch
import torch.nn.functional as fnn
from torch.autograd import Variable


def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w), 
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = fnn.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return fnn.conv2d(img, kernel, groups=n_channels)


def gaussian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = [current]

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        current = fnn.avg_pool2d(filtered, 2)
        pyr.append(current)

    return pyr