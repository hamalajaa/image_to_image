import os

import matplotlib.pyplot as plt
import matplotlib

import numpy as np
from skimage import data
from skimage.color import rgb2gray
from scipy import ndimage as ndi
from skimage import feature
import skimage.io as io

matplotlib.rcParams['font.size'] = 10


def get_canny_filter(img, gaussian_sigma):
    """
    Applies Canny filter to one image.
    """

    image = img
    # Check if image is RGB and convert to grayscale
    if image.ndim == 3:
          image = rgb2gray(image)

    edges = feature.canny(image, sigma=gaussian_sigma)
    return edges

def filter_images(urls):
    """
    Applies the Canny edge-detection filter to a list of images.
    
    Input:
        - urls      List of urls in a batch
    Output:
        - list of canny filtered images
    """

    gaussian_sigma = 3

    load_pattern = urls
    try:
        ic = io.imread_collection(load_pattern, conserve_memory=True)
        print("read images")
        filtered = []
        srcs = []
        height = 650
        width = 500
        for img in ic:
            w = img.shape[0]
            h = img.shape[1]
            print(w, h)
            img = np.pad(img, (height - h, width - w), 'constant', constant_values=(0,0))
            print("padded")
            srcs.append(img)
            filtered.append(get_canny_filter(img, gaussian_sigma))
            
    except Exception as e:
        print(e)
        print("Error occured")
        return [], []
    print("Successfully filtered images")
    return filtered, srcs

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

#edges = filter_images(urls)
#plot_paired(edges)