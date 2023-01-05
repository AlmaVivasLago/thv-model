
import os 
import argparse
import pathlib

import warnings
warnings.filterwarnings("ignore")

from thv import THV
from camera_feed import CameraFeed


def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='THV Parametrization: Process and analyze X-ray image sequences of heart valves.')
    parser.add_argument('--path', help='Path to the directory containing the files and directories to the patient data.')
    args = parser.parse_args()
    DATA_ROOT_DIR = args.path
    
    # Check if the specified directory exists
    if not os.path.isdir(DATA_ROOT_DIR):
        raise ValueError(f'Error: {DATA_ROOT_DIR} is not a valid directory.')
        
    frame_rate = 30
    save_dir = 'results'

    for path in pathlib.Path(DATA_ROOT_DIR).rglob('*'):
        if os.path.isdir(path):        
            for item in path.iterdir():
                if item.is_file():
                    image_dir = path
    
                    camera_feed = CameraFeed(frame_rate=frame_rate, image_dir=image_dir, analyzer=THV(save_dir))
                    
                    camera_feed.start()
                    camera_feed.join()
                    
                    break

if __name__ == '__main__':
    # Catch any exceptions that may be raised during execution and exit gracefully
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
        exit(1)
