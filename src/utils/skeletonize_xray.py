import numpy as np
import cv2

from skimage import filters
from skimage.util import img_as_ubyte

from utils.normalize import normalize
from utils.filter_gabor_skeleton import filter_gabor_skeleton
from utils.clean_small_connected_components import clean_small_connected_components
from utils.remove_outliers_skeleton import remove_outliers_skeleton

def skeletonize_xray(image_xray_roi, sigma=1, flag=None, merged_profile=None):
    if type(merged_profile) == type(None):
        
        kernel_sizes = [10,12, 15, 16, 18, 20, 22] 

        images_file_eq = [normalize((cv2.createCLAHE(clipLimit=limit, tileGridSize=(kernel_size, kernel_size))).apply((img_as_ubyte(image_xray_roi))))
                                  for kernel_size in kernel_sizes for limit in [5]]

        images_file_eq.append(image_xray_roi)

        sigmas = [sigma] 
        sigmas.extend(list(np.round(np.arange(0.05, 1.25, 0.25), 2)))


        profiler_searchs = [normalize(filters.meijering(image_eq, sigmas=sigmas, mode='reflect'))
                                     for cnt, image_eq in enumerate(images_file_eq)]

        profiler_searchs = np.stack(profiler_searchs, axis=0)

        profiler_searchs[profiler_searchs <= 0.15] = 0

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
    merged_profile_clean = np.where(merged_profile_clean!=0, 1.0, 0.0)
    
    merged_profile_clean_extrema = clean_small_connected_components(merged_profile_clean_extrema, flag=flag)
    merged_profile_clean_extrema[merged_profile_clean == 0.0] = 0.0

    profiler_search = {'extrema_profile': merged_profile_clean_extrema, 'merged_profile':merged_profile} # 'ends_profile': merged_profile_clean_ends

    return profiler_search