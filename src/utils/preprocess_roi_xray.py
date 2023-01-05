import numpy as np 
import cv2

from scipy import ndimage as ndi

from skimage.transform import resize
from skimage.util import img_as_ubyte

from utils.denoise_xray import denoise_xray
from utils.skeletonize_xray import skeletonize_xray
from utils.clean_small_connected_components import clean_small_connected_components

def preprocess_roi_xray(file=None, detections_results=None):                
    __, rect, ___, rect_min_r = detections_results
    
    kernelmatrix = np.ones((3, 3), np.uint8)
    kernelmatrix[:, 0] = 0
    kernelmatrix[:, -1] = 0

    bbox_slicer = sum(rect, ())
    x_l, y_l, x_u, y_u = list(np.array(list(bbox_slicer))*2)

    image_xray =  np.copy(file)
    xray_image_roi = image_xray[y_l:y_u, x_l:x_u].copy()
   
    h_file_roi, w_file_roi = xray_image_roi.shape
    check_n = True
    profilers_searchs_temp = []
    merged_profile = None
    
    xray_image_roi_denoised = denoise_xray(xray_image_roi)
    
    for flag in [True, False]:

        profiler_searchs = skeletonize_xray(xray_image_roi_denoised, sigma=1.25, flag=flag, merged_profile=merged_profile)
        merged_profile = profiler_searchs['merged_profile']
        new = np.copy(img_as_ubyte(profiler_searchs['extrema_profile']))
        labels = cv2.dilate(img_as_ubyte(profiler_searchs['extrema_profile']), kernelmatrix, iterations=3)

        if check_n:
            if np.count_nonzero(labels):
                analysis = cv2.connectedComponentsWithStats(img_as_ubyte(labels), 8)
                (totalLabels, label_ids, values, centroids) = analysis

                biggest_index = np.argmax([values[i, cv2.CC_STAT_AREA] for i  in range(1, totalLabels)])+1

                componentMask = (label_ids == biggest_index).astype("uint8") * 255

                output = np.zeros(profiler_searchs['extrema_profile'].shape, dtype="uint8")
                labels_clean = cv2.bitwise_or(output, componentMask)
                labels_clean[new==0] = 0
                profilers_searchs_temp.append(labels_clean)
            
                cnts = np.count_nonzero(labels_clean)
                if cnts > 4000:
                    check_n = False
                    break
            else:
                check_n = False
                labels_clean = clean_small_connected_components(labels, flag=False)
                profilers_searchs_temp.append(labels_clean)

    if check_n: labels_clean = np.logical_or(profilers_searchs_temp[0], profilers_searchs_temp[1])
    
    return  labels_clean