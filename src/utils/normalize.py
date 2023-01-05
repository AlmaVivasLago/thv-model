import numpy as np

def normalize(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))