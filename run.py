"""
Script to actually run the neural style transfer model and produce an output
image of any size.

@author: Riley Smith
Date: 12-30-2020
"""
import argparse
from pathlib import Path

import cv2

import torch

from data_loader import load_image, postprocess_image
from models import ImageTransformationNet

def run(cfg):
    """Runs the neural style transfer"""
    # Load image and make batch
    image = load_image(cfg['content_image'], rescale=False)
    image = image.unsqueeze(0)

    # Set device if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print alert if device is cpu
    if device == 'cpu':
        print('No GPU detected. This may take longer to run...')

    # Figure out if artist is VanGoh (VanGoh weights have slightly different shape)
    vangoh = False
    if cfg['artist'] == 'VanGoh':
        vangoh = True

    # Build network
    net = ImageTransformationNet(vangoh=vangoh).to(device)

    # Load trained weights
    artist = torch.load(f'./trained_weights/{cfg["artist"]}.pth')
    net.load_state_dict(artist['net_state_dict'])
    # net.load_state_dict(artist)

    # Put image through net
    processed_im = net(image)[0]

    # Postprocess image
    output = postprocess_image(processed_im)

    # Write output to output file
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(cfg['output_file'], output)

    print('Output image written successfully!')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_image', type=str, default='data/content/seattle.jpg',
                        help='The image to run through the neural style transfer model.')
    parser.add_argument('--artist', default='VanGoh', const='VanGoh', nargs='?',
                        choices=['VanGoh', 'Monet', 'Okeefe', 'Hokusai',
                                    'Cezanne', 'Pissarro', 'Wang'],
                        help='The artist whose style to use for the transfer.')
    parser.add_argument('--output_file', type=str, default='auto',
                        help='The file to write the output image to.')
    # Parse args
    args = parser.parse_args()

    # Turn args into dictionary
    cfg = {
        arg: getattr(args, arg) for arg in vars(args)
    }

    # Setup output file if not specified
    if cfg['output_file'] == 'auto':
        output_file = f'./results/{cfg["artist"]}.png'
        if Path(output_file).exists():
            output_file = f'./results/{cfg["artist"]}_1.png'
            i = 2
            while Path(output_file).exists():
                output_file = f'./results/{cfg["artist"]}_{i}.png'
                i += 1
        cfg['output_file'] = output_file

    run(cfg)
