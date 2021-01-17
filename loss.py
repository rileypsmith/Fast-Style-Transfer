"""
Script which uses VGG16 to setup the loss for the feed-forward neural style
transfer.

Author: Riley Smith
Date: 12-13-2020
"""
import torch
from torchvision import models
import torch.nn as nn

from utils import gram_matrix

# Load vgg model
vgg = models.vgg16(pretrained=True).features

# Set parameters to not be trainable
for p in vgg.parameters():
    p.requires_grad = False

def vgg_activations(input, vgg_net=vgg):
    """
    Functional implementation of extracting relevant style and content layers
    from VGG16 network.

    Args:
    --input: A batch of images (must have 4 dimensions) of size (bs, c, h, w)
    --vgg_net: A trained instance of vgg16

    Returns:
    The activations of the vgg network for the given input at the relevant layers.
    """
    # Identify the layers of interest
    layers_of_interest = {
        '3' : 'relu1_2',
        '8' : 'relu2_2',
        '15': 'relu3_3',
        '22': 'relu4_3'
    }

    # Create blank features dictionary to save 'checkpoints'
    features = {}

    # Sequentially step through the trained network and save activations of interest
    x = input
    for name, layer in vgg_net._modules.items():
        # Apply layer to x
        x = layer(x)
        # If this is a layer of interest, save its features
        if name in layers_of_interest:
            features[layers_of_interest[name]] = x

    # Return this feature dictionary
    return features

def perceptual_loss(features=None, style_grams=None, content_features=None,
                        cfg=None):
    """
    Computes the loss of the network, accounting for style and content loss
    (total variation loss handled separately).

    Args:
    --features: <dict> The vgg activations of the current output of the image
        transformation net
    --style_grams: <list> The gram matrix of each vgg activation for the style
        image
    --content_features: <dict> The vgg activations of the content image
    --cfg: <dict> The config dictionary

    Returns:
    <tuple> The content loss and style loss of the image transformation network.
    """
    # Content loss
    content_loss = nn.MSELoss(reduction='mean')(features['relu2_2'],
                                                content_features['relu2_2'])
    content_loss *= cfg['content_weight']

    # Style loss
    style_loss = 0
    grams = [gram_matrix(act) for act in features.values()]
    for gram, style_gram in zip(grams, style_grams):
        style_loss += nn.MSELoss(reduction='mean')(gram, style_gram)
    style_loss *= (cfg['style_weight'] / len(grams))

    return content_loss, style_loss

def total_variation_loss(batch, cfg):
    """
    Takes a batch and computes the total variation loss.

    Code for this function comes from
    https://github.com/gordicaleksa/pytorch-neural-style-transfer-fast.
    """
    batch_size = batch.shape[0]
    tv_loss = (torch.sum(torch.abs(batch[:, :, :, :-1] - batch[:, :, :, 1:])) +
            torch.sum(torch.abs(batch[:, :, :-1, :] - batch[:, :, 1:, :]))) / batch_size
    return tv_loss * cfg['tv_weight']

def get_style_grams(style_im, cfg):
    """
    Computes the style grams for the vgg activations of the given style image.

    Args:
    --style_im: <tensor> The pre-processed style image. Should have shape
        (ch, h, w)
    --cfg: <dict> The training config dictionary

    Returns:
    <list> The gram matrix for each layer in the vgg activations of the style
    image.
    """
    assert len(style_im.shape) == 3, f'Style image expected to have 3 dimensions but has {len(style_im.shape)}. Is it already a batch?'
    # First, turn the style image into a batch
    style_batch = torch.stack([style_im]*cfg['batch_size'], dim=0)

    # Get vgg activations
    style_features = vgg_activations(style_batch)

    # Get grams
    style_grams = [gram_matrix(act) for act in style_features.values()]
    return style_grams
