import numpy as np 
import cv2

from scipy import ndimage as ndi

from skimage.transform import resize
from skimage.util import img_as_ubyte

def filter_morphological_skeleton(file=None, detections_results=None, prev_data=None):
    
    file_prepared  = np.copy(file)
    ____, rect, min_rect,  rect_min_r = detections_results
    
    bbox_slicer = sum(rect, ())
    x_l, y_l, x_u, y_u = list(np.array(list(bbox_slicer))*2)
    
    kernelmatrix = np.ones((3, 3), np.uint8)
    kernelmatrix[:, 0] = 0
    kernelmatrix[:, -1] = 0

    bbox_mask = np.zeros((256, 256))

    min_rect = np.array(min_rect, dtype=np.int64).squeeze()
    value = np.where(min_rect<0)

    if len(value[0]):
        x_, y_ = value[0][0], value[1][0]
        min_rect[x_:x_+2, y_]-= min_rect[x_][y_]  

    value = np.where(min_rect<0)
    if len(value[1]):
        x_, y_ = value[0][0], value[1][0]
        min_rect[y_:, 1] -= min_rect[x_, y_] 

    min_rect = np.where(min_rect>bbox_mask.shape[1], bbox_mask.shape[1], min_rect)
    bbox_mask = cv2.drawContours(img_as_ubyte(bbox_mask), [min_rect], 0, 255, 1)
    bbox_mask = resize(bbox_mask, (512, 512))
    bbox_mask = cv2.dilate(img_as_ubyte(bbox_mask), kernelmatrix, iterations=15)

    bbox_mask = ndi.binary_fill_holes(bbox_mask)
    file_prepared[bbox_mask[y_l:y_u, x_l:x_u]==0] = 0

    output = np.zeros(file_prepared.shape, dtype="uint8")
    num_components, label_ids, stats, centroids = cv2.connectedComponentsWithStats(img_as_ubyte(file_prepared))

    min_area = 100

    for i in range(1, num_components):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            
            componentMask = (label_ids == i).astype("uint8") * 255

            output = cv2.bitwise_or(output, componentMask)
    ultimate_output=np.copy(output)
    output = cv2.dilate(img_as_ubyte(output), kernelmatrix, iterations=2)

    _countour_set, _hierarchy = cv2.findContours(img_as_ubyte(cv2.dilate(img_as_ubyte(file_prepared), kernelmatrix, iterations=3)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    _countours_sorted = sorted(_countour_set, key=lambda x: cv2.contourArea(x), reverse= True)

    _contours = _countours_sorted[0]
    _rect = cv2.minAreaRect(_contours)

    _num_components, _label_ids, _stats, _centroids = cv2.connectedComponentsWithStats(img_as_ubyte(cv2.dilate(img_as_ubyte(output), kernelmatrix, iterations=5)), connectivity = 8)

    
    _biggest_index = np.argmax([_stats[i, cv2.CC_STAT_AREA] for i  in range(1, _num_components)])+1
    w_to_print = 40

    init_x  = np.array([np.copy(_centroids)[_biggest_index][0]-w_to_print, np.copy(_centroids)[_biggest_index][0]+w_to_print])

    _rect = ((_centroids[_biggest_index][0],_centroids[_biggest_index][1]), (file_prepared.shape[0]-40, 35), (rect_min_r[2]+_rect[2])/2)
    
    _box_refined = cv2.boxPoints(_rect)
    _box_refined = np.intp(_box_refined)
    
    _box_refined = np.array(_box_refined, dtype=np.int64).squeeze()
    value = np.where(_box_refined<0)

    if len(value[0]):
        x_, y_ = value[0][0], value[1][0]
        _box_refined[x_:x_+2, y_]-= _box_refined[x_][y_]  

    value = np.where(_box_refined<0)
    if len(value[1]):
        x_, y_ = value[0][0], value[1][0]
        _box_refined[y_:, 1] -= _box_refined[x_, y_] 

    _box_refined[:, 1] = np.where(_box_refined[:, 1]>file_prepared.shape[0], file_prepared.shape[0]-2, _box_refined[:, 1])
    _box_refined[:, 0] = np.where(_box_refined[:, 0]>file_prepared.shape[1], file_prepared.shape[1]-2, _box_refined[:, 0])

    c_mask = np.zeros_like(img_as_ubyte(file_prepared))
    
    c_mask = cv2.drawContours(c_mask, [_box_refined], 0, 255, 1)
    
    c_mask = ndi.binary_fill_holes(c_mask)

    y_bbox, x_bbox = np.where(c_mask != 0)

    x_bbox, y_bbox  = np.array(x_bbox), np.array(y_bbox) 
    x_bbox_l, y_bbox_l  = list(x_bbox), list(y_bbox) 


    c_x_ymin, c_ymin =  x_bbox_l[np.where(y_bbox_l == np.min(y_bbox_l))[0][0]], np.min(y_bbox_l)
    c_xmax, c_y_xmax = np.max(x_bbox_l),  y_bbox_l[np.where(x_bbox_l == np.max(x_bbox_l))[0][0]]

    x_bbox, y_bbox = np.array([c_x_ymin, c_xmax]), np.array([c_ymin, c_y_xmax])
    A_bbox = np.vstack([x_bbox, np.ones(len(x_bbox))]).T

    m_bbox_dir, c_bbox_dir =  np.linalg.lstsq(A_bbox, y_bbox, rcond=None)[0]
    
    
    init_y  =init_x*(-1/m_bbox_dir)
    init_y += np.abs(np.sum(init_y))
    init_y +=_centroids[_biggest_index][1] - np.mean(init_y)
    
    output_clean = np.zeros(file_prepared.shape, dtype="uint8")
    
    output[c_mask!=0]=0
    for line in [0, -25, +25, -35, +35]:
         output = cv2.line(img_as_ubyte(output), (int(init_x[0])+line, int(init_y[0])+line), (int(init_x[1])+line, int(init_y[1])+line), 255, thickness=1)
    
    num_components, label_ids, stats, centroids = cv2.connectedComponentsWithStats(img_as_ubyte(output), connectivity = 8)
    
    biggest_index = np.argmax([stats[i, cv2.CC_STAT_AREA] for i  in range(1, num_components)])+1

    componentMask = (label_ids == biggest_index).astype("uint8") * 255

    output_clean = cv2.bitwise_or(output_clean, componentMask)

    __countour_set, __hierarchy = cv2.findContours(img_as_ubyte(output_clean), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    __countours_sorted = sorted(__countour_set, key=lambda x: cv2.contourArea(x), reverse= True)

    __contours = __countours_sorted[0]
    __rect = cv2.minAreaRect(__contours)
    
    __box_refined = cv2.boxPoints(__rect)
    __box_refined = np.intp(__box_refined)
    
    __box_refined = np.array(__box_refined, dtype=np.int64).squeeze()
    value = np.where(__box_refined<0)

    if len(value[0]):
        x_, y_ = value[0][0], value[1][0]
        __box_refined[x_:x_+2, y_]-= __box_refined[x_][y_]  

    value = np.where(__box_refined<0)
    if len(value[1]):
        x_, y_ = value[0][0], value[1][0]
        __box_refined[y_:, 1] -= __box_refined[x_, y_] 

    __box_refined[:, 1] = np.where(__box_refined[:, 1]>file_prepared.shape[0], file_prepared.shape[0]-5, __box_refined[:, 1])
    __box_refined[:, 0] = np.where(__box_refined[:, 0]>file_prepared.shape[1], file_prepared.shape[1]-15, __box_refined[:, 0])
            
    ultimate_mask = np.zeros_like(img_as_ubyte(file_prepared))
    
    ultimate_mask = cv2.drawContours(ultimate_mask, [__box_refined], 0, 255, 1)
    
    if np.count_nonzero(ultimate_mask) != np.count_nonzero(ndi.binary_fill_holes(ultimate_mask)):ultimate_mask = ndi.binary_fill_holes(ultimate_mask)
    else: ultimate_mask = prev_data
                        
    ultimate_output[ultimate_mask==0] = 0
    ultimate_output[file_prepared==0] = 0
    ultimate_output[c_mask!=0] = 0

    return ultimate_mask, ultimate_output
