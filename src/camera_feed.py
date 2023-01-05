import time

import cv2
import os
import queue
from threading import Thread, Lock, Condition

class ImageData:
    """
    Class representing an image data object.
    """
    def __init__(self, file_path: str, image: object) -> None:
        """
        Initializes an ImageData object with the given file path and image.

        :param file_path: The file path of the image.
        :param image: The image object.
        """
        self.file_path = file_path
        self.image = image
        
class CameraFeed:
    def __init__(self, frame_rate: int, image_dir: str,  analyzer: object): # ,
        self.frame_rate = frame_rate
        self.image_dir = image_dir
        self.analyzer = analyzer
        self.stop_flag = False
        self.queue = queue.Queue()
        self.queue_lock = Lock()
        self.queue_cv = Condition(self.queue_lock)
        self.producer_thread = Thread(target=self.produce)
        self.consumer_thread = Thread(target=self.consume)

    def produce(self):
        while not self.stop_flag:
            dir_path = self.image_dir
            file_num = 0
            file_path = dir_path / (str(file_num) + ".png")
            while os.path.exists(file_path):
                image = cv2.imread(str(file_path), 0)
                image_data = ImageData(file_path, image)
                with self.queue_lock:
                    self.queue.put(image_data)
                    self.queue_cv.notify()
                    
                time.sleep(1 / self.frame_rate)

                file_num += 1
                file_path = dir_path / (str(file_num) + ".png")
                
            self.queue.put(None)
                    
    def consume(self):
        while not self.stop_flag:
            image_count = 0
            while True:
                with self.queue_lock:
                    self.queue_cv.wait_for(lambda: not self.queue.empty())
                    image_data = self.queue.get()
                    if image_data is None:
                        self.stop_flag = True
                        self.analyzer.export_results()
                        break
                    if image_count==0:print( '\n', image_data.file_path.parent, '\n')

                    self.analyzer.run(image_data)
                    image_count += 1

    def start(self):
        self.producer_thread.start()
        self.consumer_thread.start()
    
    def join(self):

        self.producer_thread.join()
        self.consumer_thread.join()
        self.queue.task_done()

    

 