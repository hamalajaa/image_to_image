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
from torchvision import transforms

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
        
        filtered = []
        srcs = []

        # find max dimensions for padding
        #height = -1
        #width = -1
        #for img in ic:
        #    height = max(height, img.shape[0])
        #    width = max(width, img.shape[1])

        for img in ic:
            
            img = transform(img)

            # image dimensions
            #w = img.shape[1]
            #h = img.shape[0]
            #c = img.shape[2]

            ## padding size
            #padding_h = int((height - h) / 2)
            #padding_w = int((width - w) / 2)

            ## init padding matrix with c channels
            #padding = np.zeros((height, width, c))

            ## insert image to padding matrix
            #padding[padding_h:(padding_h + h):,padding_w:(padding_w + w),:] = img
            #img = padding

            srcs.append(img)
            filtered.append(get_canny_filter(img, gaussian_sigma))
            
    except Exception as e:
        print(e)
        return [], []
    
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