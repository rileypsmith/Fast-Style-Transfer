"""
Script for preprocessing images for being passed to FastStyleTransfer network
and then postprocessing back into save-able image.

Author: Riley Smith
Date: 12-13-2020
Modified: 10-15-2022
"""
import cv2
import numpy as np
from skimage.transform import resize

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

class ToTensor(object):
    """Convert ndarray to torch.tensor"""
    def __call__(self, x):
        return torch.from_numpy(x).float()

class RollAxis(object):
    """Convert from (height, width, channels) to (channels, height, width)"""
    def __call__(self, x):
        return np.rollaxis(x, 2, 0)

class BGR2RGB(object):
    """Transformation of BGR to RGB for images read with opencv"""
    def __call__(self, x):
        return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

class Rescale(object):
    """
    Class for a transformation to rescale an image to the specified size.
    """
    def __init__(self, input_shape):
        h, w, c = input_shape
        rescale_ratio = min(1, 600 / max(h, w))
        self.output_size = (int(h * rescale_ratio), int(w * rescale_ratio))

    def __call__(self, image):
        return cv2.resize(image, self.output_size, anti_aliasing=True)

def rescale(im):
    h, w, c = im.shape
    rescale_ratio = min(1, 1240 / max(h, w))
    output_size = (int(w * rescale_ratio), int(h * rescale_ratio))
    out = cv2.resize(im, output_size)
    return out

def load_image(im_path):
    """
    Load and preprocess image at given path.

    Parameters
    ----------
    im_path : str
        The path to the image to load

    Returns
    -------
    A tensor containing the normalized image.
    """
    # Make transform
    transform = transforms.Compose([
        rescale,
        BGR2RGB(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    # Read image and scale to [0,1]
    im = cv2.imread(im_path).astype(np.float32) / 255
    # Transform and return image
    return transform(im)

def postprocess_image(im):
    """
    Postprocesses an image by doing the following:
        1) Apply inverse normalization for ImageNet
        2) Roll channel dimension to be last dim
        3) Scale to range [0,255] and convert to type uint8

    Parameters
    ----------
    im : ndarray
        A float array of shape (c, h, w)

    Returns
    -------
    A uint8 array of shape (h, w, c)
    """
    # Detach gradient
    im = im.detach()

    # Reshape mean and std to un normalize
    mean = IMAGENET_MEAN.reshape(-1, 1, 1)
    std = IMAGENET_STD.reshape(-1, 1, 1)

    # Un normalize the image
    im *= std
    im += mean

    # Rescale to range [0, 255] and change to uint8
    im = (np.clip(im.numpy(), 0, 1) * 255).astype(np.uint8)

    # Roll channels axis to back
    im = np.rollaxis(im, 0, 3)

    return im
