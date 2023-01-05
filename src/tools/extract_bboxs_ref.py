import numpy as np
import cv2

from scipy import ndimage as ndi
from skimage import color

from skimage import filters, measure
from skimage.util import img_as_ubyte
from skimage.transform import resize

from utils.skeletonize_xray import skeletonize_xray
from utils.rotate_image import rotate_image

def _check_number_of_blobs(img=None):

    center = (img.shape[0] // 2, img.shape[1] // 2)

    analysis = cv2.connectedComponentsWithStats(img_as_ubyte(img), 4, cv2.CV_32S)
    (totalLabels, label_ids, values, centroids) = analysis
    
    if totalLabels != 2:
        distances = np.linalg.norm(np.array(center) - np.array(centroids[1:]), axis=1)
        closest_index = np.argmin(distances)+1
        
        componentMask = (label_ids == closest_index).astype("uint8") * 255
        
        output = np.zeros((256, 256), dtype="uint8")
        output = cv2.bitwise_or(output, componentMask)

        img[output==0] = 0

    return img 


def _evaluate_bbox_candidates(ar=None, s=None, tl=None, level=None):
    if tl<3365.23 or tl>11160:
        points = 0;
    else:
        if ar<=0.67: points = (ar/0.67) + s + level/10;
        if ar>0.67: points = (0.67/ar) + s +  level/10;
            
    return points/2 


def extract_bboxs_ref(xray_images_):
    
    kernelmatrix = np.ones((3, 3), np.uint8)

    labels_analysis, xray_images, candidate_points = [], [], []
    tracker_metadata, tracker_best_candidate = {}, []

    for filename in xray_images_:

        image_file = resize(filename, (256, 256))
 
        profiler_search = skeletonize_xray(image_file, sigma=1)

        my_frame = profiler_search['extrema_profile']

        smooth = filters.gaussian(my_frame, sigma=1.25)

        thresh_value = filters.threshold_otsu(smooth)

        thresh = smooth > thresh_value

        fill = ndi.binary_fill_holes(thresh)

        labels = measure.label(fill)

        n_unique, n_freq = np.unique(labels, return_counts=True)
        n_unique, n_freq = n_unique[1:], n_freq[1:]

        labels[labels!=n_unique[np.argmax(n_freq)]] = 0
        labels[labels != 0] = 1
        labels = cv2.dilate(img_as_ubyte(labels), kernelmatrix)
        labels[labels != 0] = 1

        labels_analysis.append(labels)
        xray_images.append(image_file)

    set_images_stack = np.dstack(tuple(labels_analysis))
    high_prob_labels =  np.sum(set_images_stack, axis=2)

    xray_bbox_mask = color.gray2rgb(xray_images[0].copy())
    
    for enum, uncertanty_level in enumerate(np.unique(high_prob_labels)[1:]):
        labels = high_prob_labels.copy()
        labels[high_prob_labels >= uncertanty_level] = 255
        labels[labels!=255] = 0

        labels_total =len(np.where(labels!=0)[0])

        countour_set, hierarchy = cv2.findContours(img_as_ubyte(labels), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        countours_sorted = sorted(countour_set, key=lambda x: cv2.contourArea(x), reverse= True)

        contours = countours_sorted[0]
        # rect = ( center (x,y), (width, height), angle of rotation )
        rect = cv2.minAreaRect(contours)
        rect_total = rect[1][0]*rect[1][1]

        box_refined = cv2.boxPoints(rect)
        box_refined = np.int0(box_refined)
        aspect_ratio = float(rect[1][1])/rect[1][0]

        tracker_metadata.update({f'>={uncertanty_level}': [labels_total, rect_total, rect[1][1], rect[1][0], aspect_ratio, labels, box_refined]})
        
    levels_range = [int(element.replace('>=','')) for element in list(tracker_metadata.keys())]
    for i in levels_range:
        labels_total, rect_total, width, height, aspect_ratio = tracker_metadata[f'>={i}'][:5]
        candidate_points.append(_evaluate_bbox_candidates(ar=aspect_ratio, s=labels_total/rect_total, tl=rect_total, level=i))
    
    
    idx_best = np.where(candidate_points==np.max(candidate_points))[0][0]
    tracker_best_candidate.append([xray_images[idx_best],  tracker_metadata[f'>={levels_range[idx_best]}'][5], candidate_points[idx_best]])
    
    xray_image, labels_, points = tracker_best_candidate[0]
    labels = np.copy(labels_)
    
    if points>=0.90:
        analysis = cv2.connectedComponentsWithStats(img_as_ubyte(labels), 8)
        (totalLabels, label_ids, values, centroids) = analysis
        
        
        countour_set, hierarchy = cv2.findContours(img_as_ubyte(labels), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS )
        countours_sorted = sorted(countour_set, key=lambda x: cv2.contourArea(x), reverse=True)

        contours = countours_sorted[0]
        rect_o = cv2.minAreaRect(contours)
        
        rect_o = (rect_o[0], (rect_o[1][0]+(rect_o[1][1]*0.2), rect_o[1][1]+(rect_o[1][1]*0.2)), rect_o[2])
        box_points_o = cv2.boxPoints(rect_o)
        box_points_o = np.int0(box_points_o)   

        xray_image_copy = np.copy(xray_image)
        
        cv2.drawContours(xray_image_copy, [box_points_o], 0, 0, 3)

    else:
        kernelmatrix = np.ones((5, 5), np.uint8)
        kernelmatrix[:, 0] = 0
        kernelmatrix[:, -1] = 0
        kernelmatrix = rotate_image(kernelmatrix, 45)

        labels = cv2.erode(img_as_ubyte(labels), kernelmatrix, iterations=2)

        h, w = labels.shape
        arft_cntrl = int(h*0.05)

        labels[:arft_cntrl, :], labels[:, :arft_cntrl]  = 0, 0
        labels[-arft_cntrl:,:], labels[:, -arft_cntrl:] = 0, 0
        
        analysis = cv2.connectedComponentsWithStats(img_as_ubyte(labels), 8)
        (totalLabels, label_ids, values, centroids) = analysis

        labels = cv2.dilate(np.copy(labels), kernelmatrix, iterations=1)

        analysis = cv2.connectedComponentsWithStats(img_as_ubyte(labels), 8)
        (totalLabels, label_ids, values, centroids) = analysis

        biggest_index = np.argmax([values[i, cv2.CC_STAT_AREA] for i  in range(1, totalLabels)])+1

        componentMask = (label_ids == biggest_index).astype("uint8") * 255

        output = np.zeros((256, 256), dtype="uint8")
        labels = cv2.bitwise_or(output, componentMask)
        
        countour_set, hierarchy = cv2.findContours(img_as_ubyte(labels), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS )
        countours_sorted = sorted(countour_set, key=lambda x: cv2.contourArea(x), reverse=True)
        
        contours = countours_sorted[0]
        rect_o = cv2.minAreaRect(contours)

        rect_o = (rect_o[0], (rect_o[1][0]+(rect_o[1][1]*0.2), rect_o[1][1]+(rect_o[1][1]*0.2)), rect_o[2])

    return [rect_o, labels, points]
        
