import numpy as np
import cv2

from scipy import ndimage as ndi
from skimage import filters

from utils.normalize import normalize
from utils.remove_outliers_skeleton import remove_outliers_skeleton
from utils.filter_gabor_skeleton import filter_gabor_skeleton
from utils.clean_small_connected_components import clean_small_connected_components

def skeletonize_xray_track(image_xray_roi,  bbox_mask_roi, sigma=1, flag=None):

    bbox_mask_fill = ndi.binary_fill_holes(bbox_mask_roi)
    
    images_file_eq = [image_xray_roi]
    kernel_sizes = [10, 15, 20]
    
    images_file_eq = np.array([normalize((cv2.createCLAHE(clipLimit=7, tileGridSize=(kernel_size, kernel_size))).apply((image_xray_roi*255).astype(np.uint8)))
                                for kernel_size in kernel_sizes])
        
    sigmas = [sigma] 
    sigmas.extend(list(np.round(np.arange(0.05, 1.5, 0.5), 2)))
    
    try:
        profiler_searchs = [filters.meijering(image_eq, sigmas=[sg], mode='reflect')
                                     for cnt, image_eq in enumerate(images_file_eq)
                                     for sg in sigmas
                                     if sg < 1.20 or cnt < 2]
    # Mac compatibility
    except:
            profiler_searchs = [filters.meijering(image_eq, sigmas=[sg], mode='reflect')
                                     for cnt, image_eq in enumerate(images_file_eq)
                                     for sg in [1, 0.05, 0.55, 1.05]
                                     if sg < 1.20 or cnt < 2]
                                    
    
                                 
                                 
    
    profiler_searchs = np.stack(profiler_searchs, axis=0)
    
    bbox_mask_roi = np.broadcast_to(bbox_mask_roi, profiler_searchs.shape)

    profiler_searchs[profiler_searchs <= 0.15] = 0
    profiler_searchs[bbox_mask_roi == 0] = 0
    
    profiler_searchs_filtered  = np.apply_over_axes(remove_outliers_skeleton, profiler_searchs, axes=(0))
    profiler_searchs = np.concatenate([profiler_searchs, profiler_searchs_filtered], axis=0)
    merged_profile = np.copy(profiler_searchs[0])

    merged_profile = np.logical_or(merged_profile, profiler_searchs[1:]).any(axis=0)

    thrld_extrema =  0.075
    
    if flag == True:
        thrld_extrema  = 0.1
        
    merged_profile_clean = clean_small_connected_components(merged_profile, flag=flag)
    merged_profile_clean = np.where(merged_profile_clean!=0, 1.0, 0.0)

    merged_profile_clean_extrema = filter_gabor_skeleton(merged_profile_clean, thrld=thrld_extrema, flag=flag)

    merged_profile_clean_extrema = clean_small_connected_components(merged_profile_clean_extrema, flag=flag)
    merged_profile_clean_extrema[merged_profile_clean == 0.0] = 0.0

    profiler_search = {'init_profile': profiler_searchs[0], 'extrema_profile': merged_profile_clean_extrema}

    return profiler_search
