import numpy as np

from skimage import filters
from skimage.morphology import disk, ball

from skimage.util import img_as_ubyte

def remove_outliers_skeleton(image, radius=1.5, threshold=0.0):
    footprint_function = disk if image.ndim == 2 else ball
    footprint = footprint_function(radius=radius)
    # Precision loss: image dtype float64 to uint8 (rank filters requirement).
    median_filtered = filters.rank.median(img_as_ubyte(image), footprint)
    outliers = (
        (image > median_filtered)
        | (image < median_filtered)
    )
    return np.where(outliers, median_filtered, image)