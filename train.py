"""
Training loop for the feed-forward neural style transfer network described by
Johnson et al. in https://arxiv.org/pdf/1603.08155.pdf%7C.

Per the license statement on Justin Johnson's GitHub, this network is free
to use for personal use, but you must contact Justin Johnson for commercial
use.

Written in PyTorch by: Riley Smith
Date: 12-13-2020
"""
import argparse
import datetime
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import build_data_loader, load_image
from loss import perceptual_loss, total_variation_loss, vgg_activations, get_style_grams
from models import ImageTransformationNet
import utils

def train(cfg):
    # Set device if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build network
    net = ImageTransformationNet().to(device)

    # Setup optimizer
    optimizer = optim.Adam(net.parameters())

    # Load state if resuming training
    if cfg['resume']:
        checkpoint = torch.load(cfg['resume'])
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])

        # Get starting epoch and batch (expects weight file in form EPOCH_<>_BATCH_<>.pt)
        parts = cfg['resume'].split('_')
        first_epoch = int(checkpoint['epoch'])
        first_batch = int(parts[-1].split('.')[0])

        # Setup dataloader
        train_data = tqdm(build_data_loader(cfg), initial=first_batch)

    else:
        # Setup dataloader
        train_data = tqdm(build_data_loader(cfg))

        # Set first epoch and batch
        first_epoch = 1
        first_batch = 0

    # Fetch style image and style grams
    style_im = load_image(cfg['style_image'], cfg)
    style_grams = get_style_grams(style_im, cfg)

    # Setup log file if specified
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    if cfg['log_file'] and not cfg['resume']:
        today = datetime.datetime.today().strftime('%m/%d/%Y')
        header = f'Feed-Forward Style Transfer Training Log - {today}'
        with open(cfg['log_file'], 'w+') as file:
            file.write(header + '\n\n')

    # Setup log CSV if specified
    if cfg['csv_log_file'] and not cfg['resume']:
        utils.setup_csv(cfg)

    for epoch in range(first_epoch, cfg['epochs'] + 1):

        # Keep track of per epoch loss
        content_loss = 0
        style_loss = 0
        total_var_loss = 0
        train_loss = 0
        num_batches = 0

        # Setup first batch to start enumerate at proper place
        if epoch == first_epoch:
            start = first_batch
        else:
            start = 0

        for i, batch in enumerate(train_data, start=start):
            batch = batch.to(device)

            # Put batch through network
            batch_styled = net(batch)

            # Get vgg activations for styled and unstyled batch
            features = vgg_activations(batch_styled)
            content_features = vgg_activations(batch)

            # Get loss
            c_loss, s_loss = perceptual_loss(features=features, content_features=content_features,
                                                style_grams=style_grams, cfg=cfg)
            tv_loss = total_variation_loss(batch_styled, cfg)
            total_loss = c_loss + s_loss + tv_loss

            # Backpropogate
            total_loss.backward()

            # Do one step of optimization
            optimizer.step()

            # Clear gradients before next batch
            optimizer.zero_grad()

            # Update summary statistics
            with torch.no_grad():
                content_loss += c_loss.item()
                style_loss += s_loss.item()
                total_var_loss += tv_loss.item()
                train_loss += total_loss.item()
                num_batches += 1

            # Update progress bar
            avg_loss = round(train_loss / num_batches, 2)
            avg_c_loss = round(content_loss / num_batches, 2)
            avg_s_loss = round(style_loss / num_batches, 1)
            avg_tv_loss = round(total_var_loss / num_batches, 3)
            train_data.set_description(f'C - {avg_c_loss} | S - {avg_s_loss} | TV - {avg_tv_loss} | Total - {avg_loss}')
            train_data.refresh()

            # Create progress image if specified
            if cfg['image_checkpoint'] and ((i + 1) % cfg['image_checkpoint'] == 0):
                save_path = str(Path(cfg['image_checkpoint_dir'], f'EPOCH_{str(epoch).zfill(3)}_BATCH_{str(i+1).zfill(5)}.png'))
                utils.make_checkpoint_image(cfg, net, save_path)

            # Save weights if specified
            if cfg['save_checkpoint'] and ((i + 1) % cfg['save_checkpoint'] == 0):
                save_path = str(Path(cfg['save_checkpoint_dir'], f'EPOCH_{str(epoch).zfill(3)}_BATCH_{str(i+1).zfill(5)}.pth'))
                checkpoint = {
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }
                torch.save(checkpoint, save_path)

            # Write progress row to CSV
            if cfg['csv_checkpoint'] and ((i + 1) % cfg['csv_checkpoint'] == 0):
                row = [epoch, i + 1, avg_c_loss, avg_s_loss, avg_tv_loss, avg_loss]
                utils.write_progress_row(cfg, row)

        # Write loss at end of each epoch
        if cfg['log_file']:
            avg_loss = round(train_loss / num_batches, 4)
            line = f'EPOCH {epoch} | Loss - {avg_loss}'
            with open(cfg['log_file'], 'a') as file:
                file.write(line + '\n')

        # Save network if specified
        if cfg['epoch_save_checkpoint'] and (epoch % cfg['epoch_save_checkpoint'] == 0):
            save_path = str(Path(cfg['save_checkpoint_dir'], f'EPOCH_{str(epoch).zfill(3)}.pth'))
            checkpoint = {
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'loss': round(train_loss / num_batches, 4)
            }
            torch.save(checkpoint, save_path)

if __name__ == '__main__':
    # Create argparser to build cfg object
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_checkpoint_dir', type=str, default='weights',
                        help='The directory to save weights to during training.')
    parser.add_argument('--save_checkpoint', type=int, default=1000,
                        help='Interval of batches to save network at. If 0, network is not saved within epochs.')
    parser.add_argument('--epoch_save_checkpoint', type=int, default=1,
                        help='Interval of epochs to save network at. If 0, network is not saved after epochs.')
    parser.add_argument('--image_checkpoint_dir', type=str, default='logs/images',
                        help='The directory to store progress images to during training.')
    parser.add_argument('--image_checkpoint', type=int, default=200,
                        help='Interval of batches to create progress image at. If 0, no progress images are created.')
    parser.add_argument('--csv_log_file', type=str, default='logs/training_log.csv',
                        help='CSV file to write training output to.')
    parser.add_argument('--csv_checkpoint', type=int, default=50,
                        help='Interval of batches to write progress line to CSV file.')
    parser.add_argument('--log_file', type=str, default='logs/training_log.txt',
                        help='The log file to write training stats to.')
    parser.add_argument('--no_log', action='store_true', help='Using this flag turns off writing stats to .txt log file during training.')
    parser.add_argument('--no_csv', action='store_true', help='Using this flag turns off writing stats to CSV during training.')
    parser.add_argument('--epochs', type=int, default=2,
                        help='The number of epochs to train for.')
    parser.add_argument('--resume', type=str, default='',
                        help='Optionally specify checkpoint to resume from.')

    parser.add_argument('--train_dir', type=str, default='data/coco',
                        help='The directory where training images are found.')
    parser.add_argument('--train_size', type=int, default=256,
                        help='The size to resize images to during training.')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size to use during training.')
    parser.add_argument('--content_weight', type=float, default=1.,
                        help='Weight for content loss.')
    parser.add_argument('--style_weight', type=float, default=4e-7,
                        help='Weight for style loss.')
    parser.add_argument('--tv_weight', type=float, default=1e-6,
                        help='Weight for total variation loss.')
    parser.add_argument('--style_image', type=str, default='data/style/van_goh.jpg',
                        help='The style image to train on.')
    parser.add_argument('--test_image', type=str, default='data/content/seattle.jpg',
                        help='The image to use for visualizing training progress.')

    # Parse args
    args = parser.parse_args()

    # Turn args into dictionary
    cfg = {
        arg: getattr(args, arg) for arg in vars(args)
    }

    # Check for 'no_log' flag
    if cfg['no_log']:
        cfg['log_file'] = None

    # Check for 'no_csv' flag
    if cfg['no_csv']:
        cfg['csv_checkpoint'] = 0

    # Run training
    train(cfg)
