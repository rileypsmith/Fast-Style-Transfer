"""
Script for loading the MS COCO dataset as a PyTorch dataloader to train the
feed forward style transfer network.

Author: Riley Smith
Date: 12-13-2020
"""
from pathlib import Path

import cv2
import numpy as np
from skimage.transform import resize

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

class CocoDataset(Dataset):
    """
    Custom dataset class for loading and pre-processing MS COCO dataset
    """
    def __init__(self, cfg):
        """
        Args:
        --cfg: <dict> The training config object
        """
        self.dir = cfg['train_dir']
        self.files = [str(fp) for fp in Path(self.dir).rglob('*.jpg')]

        self.transform = transforms.Compose([
            BGR2RGB(),
            Rescale(cfg['train_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            # RollAxis(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Read image with opencv and scale to range [0,1]
        im = cv2.imread(self.files[idx])
        im = im.astype(np.float32) / 255
        return self.transform(im)

class Rescale(object):
    """
    Class for a transformation to rescale an image to the specified size.
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        return resize(image, (self.output_size, self.output_size), anti_aliasing=True)

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

def build_data_loader(cfg):
    """
    Function to construct a CocoDataset object and build this into a PyTorch
    data loader for training.

    Args:
    --cfg: <dict> The configs dictionary

    Returns:
    A PyTorch data loader.
    """
    # Build CocoDataset
    ds = CocoDataset(cfg)

    # Make it into a dataloader
    return DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)

def load_image(im_path, cfg=None, rescale=True):
    """Load and preprocess image at given path.

    Args:
    --im_path: The path to the image to load
    --cfg: An optional dictionary of config options (not optional if rescale
        is True)
    --rescale: Whether or not to rescale the image (if False, keeps image's
        original size)

    Returns:
    A tensor containing the normalized image and optionally rescaled.
    """
    # Make transform
    if rescale:
        transform = transforms.Compose([
            BGR2RGB(),
            Rescale(cfg['train_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        transform = transforms.Compose([
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

    Args:
    --im: <ndarray> A float array of shape (c, h, w)

    Returns:
    <ndarray> A uint8 array of shape (h, w, c)
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
