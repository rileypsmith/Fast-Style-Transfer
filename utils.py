"""
Useful functions for feed forward neural style transfer.

Author: Riley Smith
Date: 12-13-2020
"""
import csv

import cv2

import torch
from torchvision import transforms

from data_loader import load_image, postprocess_image

def gram_matrix(im):
    b, c, h, w = im.size()
    im = im.view(b, c, h * w)
    output = im.bmm(im.transpose(1, 2))
    return output

def make_checkpoint_image(cfg, net, save_path):
    """
    Function to use a test content image, run it through the network, and save
    it to track training progress.

    Args:
    --cfg: <dict> The training config dictionary

    Returns:
    None. Just writes the progress image.
    """
    # Get image path for test image
    im_path = cfg['test_image']
    # Load and preprocess image
    im = load_image(im_path, cfg)
    # Turn it into a batch
    im = im.unsqueeze(0)
    # Pass it through the network
    styled = net(im)[0]
    # Post process image
    styled = postprocess_image(styled.detach())
    # Convert to BGR and write output
    styled = cv2.cvtColor(styled, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, styled)

    return

def write_progress_row(cfg, row):
    """Write training progress row to CSV file specified in config"""
    with open(cfg['csv_log_file'], 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)
    return

def setup_csv(cfg):
    """Sets up CSV log file"""
    headers = ['Epoch', 'Batch', 'Content Loss', 'Style Loss', 'TV Loss', 'Total Loss']
    with open(cfg['csv_log_file'], 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
    return
