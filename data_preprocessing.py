import os

import matplotlib.pyplot as plt
import matplotlib

import numpy as np
from skimage import data
from skimage.color import rgb2gray
from scipy import ndimage as ndi
from skimage import feature
from skimage import util
import skimage.io as io
import torch

matplotlib.rcParams['font.size'] = 10


def get_canny_filter(img, gaussian_sigma):
    """
    Applies Canny filter to one image.
    """

    image = img.detach().cpu().numpy()
    # Check if image is RGB and convert to grayscale
    if image.shape[2] == 3:
        image = rgb2gray(image)
    
    if image.ndim == 3 and image.shape[2] != 3:
        image = image[:,:,0]

    edges = feature.canny(image, sigma=gaussian_sigma)
    return edges


def filter_images(imgs):
    """
    Applies the Canny edge-detection filter to a list of images.

    Input:
        - urls      List of urls in a batch
    Output:
        - list of canny filtered images
    """

    gaussian_sigma = 3
    filtered = []

    for img in imgs:
        filtered_image = get_canny_filter(img, gaussian_sigma)

        filtered.append(torch.tensor(filtered_image))

    return filtered, imgs


def plot_paired(edges):
    """
    Plot source image and filtered image side-by-side
    """

    for original, edge in edges:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(15, 14),
                                                 sharex=True, sharey=True)
        if edge.ndim == 2:
            ax1.imshow(original, cmap=plt.cm.gray)
            ax1.axis('off')
        else:
            ax1.imshow(edge)
            ax1.axis('off')
        ax1.set_title('Original')

        ax2.imshow(edge, cmap=plt.cm.gray)
        ax2.axis('off')

        plt.show()

# edges = filter_images(urls)
# plot_paired(edges)
