import os

import numpy as np
import cv2

from skimage import  color

from skimage.util import img_as_ubyte
from skimage.transform import resize

from PIL import Image, ImageDraw

from utils.preprocess_roi_xray import preprocess_roi_xray
from utils.filter_morphological_skeleton import filter_morphological_skeleton

from tools.extract_bboxs_ref import extract_bboxs_ref
from tools.locate_structure import locate_structure

from tools.get_profile import get_profile
from tools.allocate_border import allocate_border

class ImageCollection:
    """
    Class representing a collection of ImageData objects.
    """
    def __init__(self):
        """
        Initializes an empty ImageCollection.
        """
        self.images = {}

    def add_image(self, image_data: object)-> None:
        """
        Add an image to the collection
        
        :param image_data: An object containing the image object and the file path.
        """
        self.images[image_data.file_path] = image_data.image

        
    def remove_image(self, file_path: str) -> None:
        """
        Remove an image from the collection by file path.

        :param file_path: The file path of the image to remove.
        """
        if file_path in self.images:
            del self.images[file_path]
            
    def get_images(self)-> dict:
        """
        Return the dictionary containing all images path and object.

        :return: Dict[str, ImageData] containing all the images path and object
        """
        return self.images
            
class THV(object):
    def __init__(self, save_dir):
        self.files_xray = ImageCollection()
        
        self.detections_results = {}
        self.preprocessing_results = {}
        self.intensity_profile_results = {}
        self.interpolated_extrema = {}
        
        self.extrema_coordinates = []
        self.files_xray_vis = []

        self.shape_check = None
        self.temp_kp = [0, 0]
        
        self.save_dir = save_dir
        self.parent_dir = None
        
        self.refs_init = False 
        self.bboxs_ref = None
        
        self.abort = False
        
        self.best_bbox_check  = None
        self.check_init_bbox = True

    #TOREFACTOR - (Not DRY)
    def run(self, file)-> None:
        self.files_xray.add_image(file)
        self.files_xray_vis.append(np.copy(file.image))
        if not self.parent_dir: self.parent_dir = file.file_path.parent
            
        if len(self.files_xray.get_images()) == 5 and not self.refs_init:
            self._find(file_list=self.files_xray.get_images().values(), tracker_init=self.refs_init)
            self._find(file_list=self.files_xray.get_images(), tracker_init=True)
        
            if len(self.detections_results):    
                self._compute_profile(file_list=self.files_xray.get_images(), detections_results=self.detections_results)
            
                self._measure(file_list=self.files_xray.get_images(), detections_results=self.detections_results, border_init=self.refs_init)
                
                self.refs_init = True 

        if self.refs_init and not self.abort:
            self._find(file_list=self.files_xray.get_images(), tracker_init=self.refs_init)
        
            if len(self.detections_results):    
                self._compute_profile(file_list=self.files_xray.get_images(), detections_results=self.detections_results)
                self._measure(file_list=self.files_xray.get_images(), detections_results=self.detections_results, border_init=self.refs_init)
                
    def export_results(self):
        self._save_results()
        
    
    def _find(self, file_list=None, tracker_init=None)-> None:
        if tracker_init == False or None:
            self.bboxs_ref = extract_bboxs_ref(file_list)
        else:
            rect_o, labels, points = self.bboxs_ref
            for file_ref, file in file_list.items():
                ref_loc = int(os.path.splitext(os.path.split(file_ref)[-1])[0])
                if not self.detections_results.__contains__(ref_loc):
                    if ref_loc==0: prev_data = None
                    else: prev_data = self.detections_results[ref_loc-1] 
                    
                    center, rect, min_rect,  rect_min_r = locate_structure(file=file, ref_loc=ref_loc, bbox_detection=self.bboxs_ref, prev_data=prev_data)
                    
                    self.detections_results.update({ref_loc: [center, rect, min_rect,  rect_min_r]})

    
    def _compute_profile(self, file_list=None, detections_results=None)-> None:
        for check_counter, (file_ref, file) in enumerate(file_list.items()):
            ref_loc = int(os.path.splitext(os.path.split(file_ref)[-1])[0])

            if not self.intensity_profile_results.__contains__(ref_loc):
                if not len(self.intensity_profile_results): 
                    prev_data = None
                    file_roi_preprocessed = preprocess_roi_xray(file=file, detections_results=self.detections_results[ref_loc])
                    best_roi_bbox, search_roi_image = filter_morphological_skeleton(file=file_roi_preprocessed, detections_results=self.detections_results[ref_loc], prev_data=prev_data)
                    self.best_bbox_check  = best_roi_bbox
                    while not np.count_nonzero(self.best_bbox_check) and self.check_init_bbox:
                        check_counter += 1

                        file_roi_preprocessed = preprocess_roi_xray(file=list(file_list.values())[check_counter], detections_results=self.detections_results[check_counter])

                        best_roi_bbox, search_roi_image = filter_morphological_skeleton(file=file_roi_preprocessed, detections_results=self.detections_results[check_counter], prev_data=prev_data)

                        self.best_bbox_check  = best_roi_bbox

                    else:
                        intensity_profile = get_profile(search_roi_image=search_roi_image, best_roi_bbox=self.best_bbox_check)
                        self.intensity_profile_results.update({check_counter: [best_roi_bbox, search_roi_image, intensity_profile]})

                        for update_file in range(check_counter-1):
                            file_roi_preprocessed = preprocess_roi_xray(file=list(file_list.values())[update_file], detections_results=self.detections_results[update_file])
                            _, search_roi_image = filter_morphological_skeleton(file=file_roi_preprocessed, detections_results=self.detections_results[ref_loc], prev_data=prev_data)

                            intensity_profile = get_profile(search_roi_image=search_roi_image, best_roi_bbox=self.best_bbox_check)
    
                            self.intensity_profile_results.update({update_file: [best_roi_bbox, search_roi_image, intensity_profile]})
                            
                            self.check_init_bbox=False
                else: 
                    prev_data = list(self.intensity_profile_results.values())[-1][0]
                    file_roi_preprocessed = preprocess_roi_xray(file=file, detections_results=self.detections_results[ref_loc])
                    best_roi_bbox, search_roi_image = filter_morphological_skeleton(file=file_roi_preprocessed, detections_results=self.detections_results[ref_loc], prev_data=prev_data)
                    
                    intensity_profile = get_profile(search_roi_image=search_roi_image, best_roi_bbox=best_roi_bbox)
                    self.intensity_profile_results.update({ref_loc: [best_roi_bbox, search_roi_image, intensity_profile]})

    def _measure(self, file_list=None, detections_results=None, border_init=None)-> None:
        if border_init == False or None:
            for file_ref, file in file_list.items():
                ref_loc = int(os.path.splitext(os.path.split(file_ref)[-1])[0])

                center, rect, min_rect,  rect_min_r = self.detections_results[ref_loc]
                intensity_profile = self.intensity_profile_results[ref_loc][2]
                    
                interpolated_extrema = allocate_border(intensity_profile['full_profile'], intensity_profile['index'], rect_min_r[1][1]*2)
                self.interpolated_extrema.update({ref_loc:interpolated_extrema})
        
            check = np.array([len(intepolated) for intepolated in list(self.interpolated_extrema.values())])
            
            if (check>0).all(): 
                for file_ref, file in file_list.items():
                    ref_loc = int(os.path.splitext(os.path.split(file_ref)[-1])[0])
                    
                    center, rect, min_rect,  rect_min_r = self.detections_results[ref_loc]
                    bbox_slicer = sum(rect, ())

                    x_l, y_l, x_u, y_u = list(np.array(list(bbox_slicer))*2)

                    self.interpolated_extrema[ref_loc]['left_filtered'][0, :] +=x_l
                    self.interpolated_extrema[ref_loc]['left_filtered'][1, :] += y_l
                    
                    self.interpolated_extrema[ref_loc]['right_filtered'][0, :] +=x_l
                    self.interpolated_extrema[ref_loc]['right_filtered'][1, :] +=y_l
                
                    self.extrema_coordinates.append([self.interpolated_extrema[ref_loc]['left_filtered'], self.interpolated_extrema[ref_loc]['right_filtered']])

                    xray_image = color.gray2rgb(file)
                    xray_image = Image.fromarray(xray_image)
            
                    draw = ImageDraw.Draw(xray_image)
                    
                    draw.line(list(zip(self.interpolated_extrema[ref_loc]['right_filtered'][0, :], self.interpolated_extrema[ref_loc]['right_filtered'][1, :])), fill=(0, 0, 255), width=1)
                    draw.line(list(zip(self.interpolated_extrema[ref_loc]['left_filtered'][0, :], self.interpolated_extrema[ref_loc]['left_filtered'][1, :])), fill=(0, 0, 255), width=1)

                    xray_image = np.array(xray_image)
                    self.files_xray_vis.append(xray_image)

                    save_path = os.path.join(self.save_dir, file_ref)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    cv2.imwrite(save_path, xray_image)                    
                    print("Saved image: ", save_path)
                    
            else: self.abort = True
        
        if border_init and not self.abort:        
            for file_ref, file in file_list.items():
                ref_loc = int(os.path.splitext(os.path.split(file_ref)[-1])[0])
                if not self.interpolated_extrema.__contains__(ref_loc):
    
                    center, rect, min_rect,  rect_min_r = self.detections_results[ref_loc]
                    intensity_profile = self.intensity_profile_results[ref_loc][2]

                    interpolated_extrema = allocate_border(intensity_profile['full_profile'], intensity_profile['index'], rect_min_r[1][1]*2)
                    bbox_slicer = sum(rect, ())

                    x_l, y_l, x_u, y_u = list(np.array(list(bbox_slicer))*2)
            
                    if len(interpolated_extrema) !=4:
                        self.abort = True
                        break
                    else:
                        self.interpolated_extrema.update({ref_loc:interpolated_extrema})
                    
                        self.interpolated_extrema[ref_loc]['left_filtered'][0, :] +=x_l
                        self.interpolated_extrema[ref_loc]['left_filtered'][1, :] += y_l
                        
                        self.interpolated_extrema[ref_loc]['right_filtered'][0, :] +=x_l
                        self.interpolated_extrema[ref_loc]['right_filtered'][1, :] +=y_l

                        self.extrema_coordinates.append([self.interpolated_extrema[ref_loc]['left_filtered'], self.interpolated_extrema[ref_loc]['right_filtered']])

                        xray_image = color.gray2rgb(file)
                        xray_image = Image.fromarray(xray_image)
                            
                        draw = ImageDraw.Draw(xray_image)
                        
                        draw.line(list(zip(self.interpolated_extrema[ref_loc]['right_filtered'][0, :], self.interpolated_extrema[ref_loc]['right_filtered'][1, :])), fill=(0, 0, 255), width=1)
                        draw.line(list(zip(self.interpolated_extrema[ref_loc]['left_filtered'][0, :], self.interpolated_extrema[ref_loc]['left_filtered'][1, :])), fill=(0, 0, 255), width=1)
                     
                        xray_image = np.array(xray_image)
                        self.files_xray_vis.append(xray_image)

                        save_path = os.path.join(self.save_dir, file_ref)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(save_path, xray_image)
                        print("Saved image: ", save_path)


                        
    def _save_results(self):
        if len(self.extrema_coordinates):
            with open(os.path.join(self.save_dir, self.parent_dir, "extrema_coordinates.npy"), 'wb') as c:
                np.save(c, np.array(self.extrema_coordinates))

        if len( self.files_xray_vis):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(os.path.join(self.save_dir, self.parent_dir, 'overplot_sequence.mp4'), fourcc, 20, (512, 512))
                    
            for frame in self.files_xray_vis:
                frame = resize(frame, (512, 512))
                out.write(img_as_ubyte(frame))
                
            out.release()
