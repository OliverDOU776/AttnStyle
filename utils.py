import os
import sys

def setup_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clipstyler_dir = os.path.join(script_dir, "CLIPstyler")
    diffclf_dir = os.path.join(script_dir, "diffusion-classifier")
    if clipstyler_dir not in sys.path:
        sys.path.append(clipstyler_dir)
    if diffclf_dir not in sys.path:
        sys.path.append(diffclf_dir)

setup_paths()

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

##############################################################################
# Image Loading / Normalization
##############################################################################
def load_image2(img_path, img_height=None, img_width=None):
    """
    Loads an image from disk, optionally resizes to (img_width, img_height).
    Returns a torch.Tensor of shape (1,3,H,W) in [0,1].
    """
    image = Image.open(img_path).convert('RGB')
    if (img_width is not None) and (img_height is not None):
        image = image.resize((img_width, img_height))
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)  # (1,3,H,W)
    return image

def img_normalize(image, device):
    """
    Normalizes an image using ImageNet means, as typically done for VGG.
    Expects input shape (B,3,H,W) in [0,1], returns a normalized tensor.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1,-1,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1,-1,1,1)
    return (image - mean) / std

def img_denormalize(image, device):
    """
    Inverse of img_normalize. 
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1,-1,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1,-1,1,1)
    return image * std + mean

def im_convert2(tensor):
    """
    Convert a normalized or unnormalized image tensor to [0,1] numpy array (H,W,3).
    Expects shape (1,3,H,W).
    """
    image = tensor.detach().cpu().clone().squeeze(0).numpy()
    image = np.transpose(image, (1,2,0))  # (H,W,3)
    image = np.clip(image, 0, 1)
    return image

##############################################################################
# VGG Feature Extraction
##############################################################################
def get_features(image, model, layers=None):
    """
    Passes `image` through `model` (VGG19), returning a dict of features.
    By default, uses these layers:
      {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10':'conv3_1',
        '19':'conv4_1',
        '21':'conv4_2',
        '28':'conv5_1',
        '31':'conv5_2'
      }
    """
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10':'conv3_1',
            '19':'conv4_1',
            '21':'conv4_2',
            '28':'conv5_1',
            '31':'conv5_2'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

##############################################################################
# High-Level Helpers
##############################################################################
def load_content_image(content_path, img_height, img_width, device):
    content_image = load_image2(content_path, img_height=img_height, img_width=img_width)
    content_image = content_image.to(device)
    return content_image

def get_result_path(args, suffix):
    """
    Generates a path for saving results under 'results/<suffix>/<args.text>'
    """
    base_dir = "results"
    style_dir = os.path.join(base_dir, suffix, args.text)
    return style_dir
