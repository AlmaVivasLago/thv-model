import numpy as np

from scipy import ndimage as ndi
from skimage import measure

def clean_small_connected_components(processed_image, flag=None, ends=None):
    n_freq_thr = 30
    
    if flag == True:
        n_freq_thr = 50 
        
    if ends == True:
        n_freq_thr = 200
        
    processed_image_clean =  processed_image.copy()  
    processed_image_bin =  processed_image_clean.copy()  

    processed_image_bin[processed_image_bin != 0] = 1
    fill = ndi.binary_fill_holes(processed_image_bin)
    
    labels = measure.label(fill)
    
    n_unique, n_freq = np.unique(labels, return_counts=True)
    
    noise = np.where(n_freq<=n_freq_thr)[0]
    
    for n in noise:
        labels[labels == n] = 0
        
    processed_image_clean[labels == 0] = 0    
    
    return processed_image_clean