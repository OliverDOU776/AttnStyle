import utils
utils.setup_paths()

import torch
import torch.nn as nn
from torchvision import models
from argparse import Namespace
from diffusion.models import get_sd_model  # Adjust if your DMD2 code entry point differs

##############################################################################
# 1. Load a pretrained VGG model for content loss
##############################################################################
def load_vgg_model(device):
    """
    Loads a pretrained VGG19 (feature layers only), frozen for content loss.
    """
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad_(False)
    return vgg

##############################################################################
# 2. Initialize a single/few-step diffusion model from DMD2 (or similar).
##############################################################################
def initialize_diffusion_model(args, device):
    """
    Initializes the diffusion model components (VAE, text encoder, U-Net, etc.).
    This can be replaced or adjusted to load a single/few-step pretrained model
    from DMD2 if you have a custom DMD2 loader.
    """
    model_args = Namespace(
        img_size=args.img_height,
        batch_size=args.batch_size,
        dtype=args.dtype,
        loss=args.loss,
        version=args.version,
        diff_times=args.diff_times,
        device=device,
    )
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(model_args)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    return vae, tokenizer, text_encoder, unet, scheduler

##############################################################################
# 3. AttnStyle U-Net Architecture
##############################################################################

class CrossAttentionBlock(nn.Module):
    """
    A simple cross-attention block that takes:
      - Query = stylized features
      - Key/Value = content features
    and outputs an updated stylized feature. 
    """
    def __init__(self, in_dim):
        super().__init__()
        self.query_proj = nn.Linear(in_dim, in_dim)
        self.key_proj   = nn.Linear(in_dim, in_dim)
        self.value_proj = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, styl_features, cont_features):
        """
        styl_features: (B, C, H*W) or (B, C) flattened
        cont_features: (B, C, H*W) or (B, C) flattened
        """
        # For simplicity, assume shapes are (B, C, N)
        Q = self.query_proj(styl_features.transpose(1,2))  # -> (B, N, C)
        K = self.key_proj(cont_features.transpose(1,2))    # -> (B, N, C)
        V = self.value_proj(cont_features.transpose(1,2))  # -> (B, N, C)
        
        # Compute attention
        attn_scores = torch.bmm(Q, K.transpose(1,2)) / (Q.size(-1) ** 0.5)  # (B, N, N)
        attn_weights = self.softmax(attn_scores)
        out = torch.bmm(attn_weights, V)  # (B, N, C)
        
        # Project back
        out = out.transpose(1,2)  # (B, C, N)
        return out

class AttnStyleUNet(nn.Module):
    """
    A lightweight U-Net-based encoder-decoder with cross-attention modules in 
    the decoder to align stylized and content features.
    """
    def __init__(self):
        super().__init__()
        # ---------------- Encoder ----------------
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # ---------------- Decoder ----------------
        self.dec_deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        
        # Cross-attention blocks (simple example)
        self.cross_attn1 = CrossAttentionBlock(256)
        self.cross_attn2 = CrossAttentionBlock(128)
        self.cross_attn3 = CrossAttentionBlock(64)
        
        self.relu = nn.ReLU(inplace=True)
        
    def encode(self, x):
        """
        Encoder forward pass, returns intermediate features for cross-attn usage.
        """
        e1 = self.relu(self.enc_conv1(x))  # (B,64,H/2,W/2)
        e2 = self.relu(self.enc_conv2(e1)) # (B,128,H/4,W/4)
        e3 = self.relu(self.enc_conv3(e2)) # (B,256,H/8,W/8)
        return e1, e2, e3
    
    def decode(self, feats_enc, feats_cont):
        """
        Decoder forward pass with cross-attn. 
        feats_enc: tuple of encoder features (e1, e2, e3)
        feats_cont: tuple of content features from the encoder pass of content image
        """
        e1, e2, e3 = feats_enc
        c1, c2, c3 = feats_cont
        
        # Cross-attn input shape is (B, C, H*W). Flattening:
        B, C, H, W = e3.shape
        styl_flat = e3.view(B, C, -1)
        cont_flat = c3.view(B, C, -1)
        e3_attn = self.cross_attn1(styl_flat, cont_flat)
        e3_attn = e3_attn.view(B, C, H, W)
        
        d1 = self.relu(self.dec_deconv1(e3_attn))  # up to (B,128,H/4,W/4)
        # Next cross-attn
        B, C, H, W = d1.shape
        styl_flat = d1.view(B, C, -1)
        cont_flat = c2.view(B, C, -1)
        d1_attn = self.cross_attn2(styl_flat, cont_flat).view(B, C, H, W)
        
        d2 = self.relu(self.dec_deconv2(d1_attn))  # up to (B,64,H/2,W/2)
        # Next cross-attn
        B, C, H, W = d2.shape
        styl_flat = d2.view(B, C, -1)
        cont_flat = c1.view(B, C, -1)
        d2_attn = self.cross_attn3(styl_flat, cont_flat).view(B, C, H, W)
        
        d3 = self.dec_deconv3(d2_attn)  # up to (B,3,H,W)
        # Sigmoid or clamp to produce output
        out = torch.sigmoid(d3)
        return out
    
    def forward(self, content_img, content_feats=None):
        """
        content_img: the content image to be stylized
        content_feats: if provided, we use them as the 'content side' for cross-attn
                       if not provided, we first encode the same image to get them
        """
        # If content feats are not given, encode the same image 
        if content_feats is None:
            c1, c2, c3 = self.encode(content_img)
            content_feats = (c1, c2, c3)
        
        # "Stylization" pass: we just re-encode the content, then decode with cross-attn
        e1, e2, e3 = self.encode(content_img)
        styl_feats = (e1, e2, e3)
        
        output = self.decode(styl_feats, content_feats)
        return output, styl_feats, content_feats
