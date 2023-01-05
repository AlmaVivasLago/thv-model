import numpy as np
import cv2

from scipy import ndimage as ndi
from skimage import filters, measure
from skimage.util import img_as_ubyte

from skimage.transform import resize

from utils.rotate_image import rotate_image
from utils.skeletonize_xray_track import skeletonize_xray_track

def locate_structure(file=None, ref_loc=None, bbox_detection=None, prev_data=None):

    bbox_test, template_, points = bbox_detection
    Y_R, H_R, X_R, W_R, angle = bbox_test[0][1], bbox_test[1][0], bbox_test[0][0], bbox_test[1][1],bbox_test[2] 
    H_R, W_R = H_R+(H_R*0.3), W_R+(W_R*0.3)

    box_refined = cv2.boxPoints(bbox_test)
    box_refined = np.intp(box_refined)
    box_refined = np.where(box_refined<0, 0, box_refined)

    x_l, y_l  = np.round(np.min(box_refined[:, 0]), 0), np.round(np.min(box_refined[:, 1]), 0)
    x_u, y_u = np.round(np.max(box_refined[:, 0]), 0) , np.round(np.max(box_refined[:, 1]), 0)
    
    x_l, y_l = int(x_l-(W_R*0.15)), int(y_l-(H_R*0.15))
    x_u, y_u = int(x_u+(W_R*0.15)), int(y_u+(H_R*0.15))
    
    slicer = [x_l, y_l, x_u, y_u]
    slicer = np.where(np.array(slicer)<0, 0, np.array(slicer))
    
    x_l, y_l, x_u, y_u = list(slicer)
    if prev_data: center_p, rect_p, min_rect_p, rect_min_r_p =  prev_data
    else: center_p = [X_R, Y_R]

    image_file = resize(file, (file.shape[0]//2, file.shape[0]//2))
     
    profiler_search = skeletonize_xray_track(image_file,  np.ones_like(image_file), sigma=1)
    
    my_frame = profiler_search['extrema_profile']
    
    smooth = filters.gaussian(my_frame, sigma=1.5)
    
    thresh_value = filters.threshold_otsu(smooth)
    
    thresh = smooth > thresh_value
    
    fill = ndi.binary_fill_holes(thresh)
    
    labels = measure.label(fill)
    
    h, w = labels.shape
    arft_cntrl = int(h*0.05)

    labels[:arft_cntrl, :], labels[:, :arft_cntrl]  = 0, 0
    labels[-arft_cntrl:,:], labels[:, -arft_cntrl:] = 0, 0

    labels_r = np.copy(labels)
    
    if points<0.80:
        kernelmatrix = np.ones((5, 5), np.uint8)
        kernelmatrix[:, 0] = 0
        kernelmatrix[:, -1] = 0
        
        kernelmatrix = rotate_image(kernelmatrix, 45)

        labels = cv2.erode(img_as_ubyte(labels), kernelmatrix, iterations=1)

        analysis = cv2.connectedComponentsWithStats(img_as_ubyte(labels), 8)
        (totalLabels, label_ids, values, centroids) = analysis

        labels = cv2.dilate(img_as_ubyte(labels), kernelmatrix, iterations=1)

        analysis = cv2.connectedComponentsWithStats(img_as_ubyte(labels), 8)
        (totalLabels, label_ids, values, centroids) = analysis

        biggest_index = np.argmax([values[i, cv2.CC_STAT_AREA] for i  in range(1, totalLabels)])+1
        
        componentMask = (label_ids == biggest_index).astype("uint8") * 255

        output = np.zeros((256, 256), dtype="uint8")
        labels = cv2.bitwise_or(output, componentMask)
        
    else:
        n_unique, n_freq = np.unique(labels, return_counts=True)
        n_unique, n_freq = n_unique[1:], n_freq[1:]

        labels[labels!=n_unique[np.argmax(n_freq)]] = 0
        labels[labels != 0] = 1
    
    template = np.copy(template_)
    
    if ref_loc<5:

        big_bbox_mask = np.zeros_like(img_as_ubyte(image_file))
        big_bbox_mask = cv2.rectangle(big_bbox_mask, (x_l, y_l), (x_u, y_u), 255, 0, 1)
        big_bbox_mask = ndi.binary_fill_holes(big_bbox_mask)
 
        labels[big_bbox_mask==0] = 0
        center_x, center_y = X_R, Y_R
        rect = [(x_l, y_l), (x_u, y_u)]

    else:   
        query = img_as_ubyte(image_file).copy()

        w, h, = template[y_l:y_u, x_l:x_u].shape[0], template[y_l:y_u, x_l:x_u].shape[1]
        
        big_bbox_mask = np.zeros_like(img_as_ubyte(image_file))
        if len(min_rect_p)!=1:
            min_rect_p = [min_rect_p]
            
        big_bbox_mask = cv2.drawContours(big_bbox_mask,min_rect_p ,  0, 255, 1)
        big_bbox_mask = ndi.binary_fill_holes(big_bbox_mask)
        labels[big_bbox_mask==0] = 0

        query[labels!=0] = 255

        res = cv2.matchTemplate(query, img_as_ubyte(template)[y_l:y_u, x_l:x_u], cv2.TM_CCORR_NORMED)

        minv, maxv, cmin, cmax = cv2.minMaxLoc(res)
       
        y, x = np.unravel_index(np.argmax(res), res.shape)
        center_x, center_y = x+(h/2), y+(w/2)
        
    if np.count_nonzero(labels)<200:labels=np.copy(labels_r)

    analysis = cv2.connectedComponentsWithStats(img_as_ubyte(labels), 8)
    (totalLabels, label_ids, values, centroids) = analysis
    biggest_index = np.argmax([values[i, cv2.CC_STAT_AREA] for i  in range(1, totalLabels)])+1

    if (centroids !=None).any():
        if ref_loc!=0:
            x_o, y_o = center_p[0]-centroids[biggest_index][0], center_p[1]-centroids[biggest_index][1]
        else:
            x_o, y_o = center_x-centroids[biggest_index][0], center_y-centroids[biggest_index][1]

        if (np.abs(np.array([x_o, y_o]))>15).any() and ref_loc>5:
    
            rect = [rect_p[0], rect_p[1]]
        
            rect_o_ = ((center_p[0], center_p[1]), (H_R, W_R), angle)
            rect_min_r = rect_o_
            box_points_o_ = cv2.boxPoints(rect_o_)
            box_points_o_ = np.intp(box_points_o_)   
            min_rect = [box_points_o_]
            
            center_x, center_y = center_p[0], center_p[1]
            center = [center_x, center_y]

        else:
            if (np.abs(np.array([x_o, y_o]))>5).any() and np.count_nonzero(labels)>4000:
                if np.abs(y_o)>10:y_o = 2*np.sign(y_o)
                if np.abs(x_o)>15:x_o = 8*np.sign(x_o)
    
                rect_o_ = ((center_x-x_o, center_y-y_o), (H_R, W_R), angle)
                rect_min_r = rect_o_
    
                box_points_o_ = cv2.boxPoints(rect_o_)
                box_points_o_ = np.intp(box_points_o_)   
                min_rect = box_points_o_
                rect = [(x_l, y_l), (x_u, y_u)]
                center = [center_x-x_o, center_y-y_o]
            else:
                rect_o_ = ((center_x, center_y), (H_R, W_R), angle)
                rect_min_r = rect_o_
    
                box_points_o_ = cv2.boxPoints(rect_o_)
                box_points_o_ = np.intp(box_points_o_)   
    
                min_rect = box_points_o_
                
                rect = [(x_l, y_l), (x_u, y_u)]
                center = [center_x, center_y] 

    return center, rect, min_rect,  rect_min_r