"""
Script to actually run the neural style transfer model and produce an output
image of any size.

@author: Riley Smith
Date: 12-30-2020
"""
import argparse
from pathlib import Path
import uuid

import cv2

import torch

from .data_loader import load_image, postprocess_image
from .models import ImageTransformationNet

ROOT = str(Path(__file__).absolute().parent.parent)

def run(cfg):
    """Runs the neural style transfer"""
    # Load image and make batch
    image = load_image(str(Path(ROOT, cfg['image_path'].strip('/'))))
    image = image.unsqueeze(0)

    # Set device if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build network
    net = ImageTransformationNet(vangoh=cfg['vangoh']).to(device)

    # Load trained weights
    artist = torch.load(str(Path(ROOT, cfg['weight_path'])))
    net.load_state_dict(artist['net_state_dict'])

    # Put image through net
    processed_im = net(image)[0]

    # Postprocess image
    out_im = postprocess_image(processed_im)

    # Write output to output file
    out_im = cv2.cvtColor(out_im, cv2.COLOR_RGB2BGR)
    out_path = str(Path(ROOT, 'media', f'{uuid.uuid4()}.jpg'))
    cv2.imwrite(out_path, out_im)

    return out_path
