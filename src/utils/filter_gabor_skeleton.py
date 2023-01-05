import numpy as np

from skimage.filters import gabor_kernel
from scipy.signal import convolve2d
from skimage import img_as_float

def filter_gabor_skeleton(skeleton_image, thrld=None, flag=None):
    
    skeleton_image_bin = np.where(skeleton_image.copy() != 0, 1.0, 0.0)
    skeleton_image_bin = img_as_float(skeleton_image_bin)

    theta = np.arange(np.pi/16*11, np.pi-np.pi/16, np.pi/18)
    ffs = np.round(np.arange(0.1, 0.5, 0.1), 1)

    thetas = np.zeros(len(theta) * len(ffs))
    filtered = np.zeros((len(theta) * len(ffs), skeleton_image_bin.shape[0], skeleton_image_bin.shape[1]))

    for i, ff in enumerate(ffs):
        for j, t in enumerate(theta):
            g = gabor_kernel(frequency=ff, theta=t)
            filt = convolve2d(skeleton_image_bin, g, mode='same')

            thetas[i * len(ffs) + j] = t
            filtered[i * len(ffs) + j] = np.real(filt)

    filtered = np.where(filtered <= thrld, 0, 1)

    merged_profile = np.logical_or.reduce(filtered, axis=0)

    return merged_profile

