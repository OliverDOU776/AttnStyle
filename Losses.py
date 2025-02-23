import utils
utils.setup_paths()

import torch
import torch.nn.functional as F
from diffusion.models import get_sd_model
from torchvision.transforms.functional import InterpolationMode
from argparse import Namespace

################################################################################
# 1. Classification Loss (Global) using single/few-step pretrained diffusion
################################################################################
def classification_loss(stylized_img, text_prompt, model_args):
    """
    This implements the global classification loss L_cls using a single/few-step
    diffusion model (from DMD2) as a zero-shot classifier.
    
    The model tries to see if stylized_img is recognized as matching text_prompt
    by measuring the noise prediction error at random timesteps.
    """
    device = model_args.device
    # Convert stylized_img to the right shape/dtype
    if stylized_img.dim() == 3:
        stylized_img = stylized_img.unsqueeze(0)
    stylized_img = stylized_img.to(device, dtype=torch.float32)
    
    # Load or re-use stable/diffusion model
    # (We assume model_args has .vae, .tokenizer, .text_encoder, .unet, .scheduler, 
    #  or a function to retrieve them. Adapt as needed.)
    vae, tokenizer, text_encoder, unet, scheduler = model_args.vae, model_args.tokenizer, model_args.text_encoder, model_args.unet, model_args.scheduler
    
    # Encode image into latent space
    with torch.no_grad():
        x0 = vae.encode(stylized_img).latent_dist.mean
        x0 *= 0.18215  # typical scale in stable diffusion
    
    # Encode the text
    text_input = tokenizer(
        [text_prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids)[0]
    
    # Compute the MSE between predicted noise and actual noise across random timesteps
    T = scheduler.config.num_train_timesteps
    n_samples = getattr(model_args, 'diff_times', 1)
    total_loss = 0.0
    for _ in range(n_samples):
        t = torch.randint(0, T, (1,), device=device).long()
        noise = torch.randn_like(x0)
        
        alpha_cumprod = scheduler.alphas_cumprod[t].view(-1,1,1,1).to(device)
        sqrt_alpha_cumprod = alpha_cumprod.sqrt()
        sqrt_one_minus_alpha_cumprod = (1 - alpha_cumprod).sqrt()
        x_t = sqrt_alpha_cumprod*x0 + sqrt_one_minus_alpha_cumprod*noise
        
        # Predict the noise
        with torch.no_grad():
            noise_pred = unet(x_t, t, encoder_hidden_states=text_embeddings).sample
        
        loss = F.mse_loss(noise_pred, noise)
        total_loss += loss
    
    return total_loss / n_samples

################################################################################
# 2. Multi-Scale Patch Classification Loss
################################################################################
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

def multi_scale_classification_loss(stylized_img, text_prompt, model_args):
    """
    Enforces stylistic consistency at multiple scales by extracting patches
    from the stylized image at different resolutions and applying the same
    classification loss as above to each patch.
    
    For demonstration, we show global, intermediate (1/2 resolution), 
    and local (128x128 patches).
    """
    device = model_args.device
    
    # Global scale
    L_global = classification_loss(stylized_img, text_prompt, model_args)
    
    # Intermediate scale (downsample x2)
    intermediate = F.interpolate(stylized_img, scale_factor=0.5, mode='bilinear', align_corners=False)
    L_intermediate = classification_loss(intermediate, text_prompt, model_args)
    
    # Local scale (crop random 128x128 if stylized is bigger)
    # For robust training, you might do multiple random patches and average them
    # For demonstration, we'll do one random crop
    b, c, h, w = stylized_img.shape
    crop_size = min(128, h, w)
    if crop_size < 128:
        # if image is smaller than 128, skip local scale
        L_local = 0.0
    else:
        top = torch.randint(0, h-crop_size+1, (1,)).item()
        left = torch.randint(0, w-crop_size+1, (1,)).item()
        local_patch = stylized_img[:, :, top:top+crop_size, left:left+crop_size]
        L_local = classification_loss(local_patch, text_prompt, model_args)
    
    # Weighted sum
    # You could define separate lambdas or an internal weighting
    # For demonstration, we do a simple average
    ms_loss = (L_global + L_intermediate + L_local) / 3.0
    return ms_loss

################################################################################
# 3. Attention-Based Alignment Loss
################################################################################
def attention_alignment_loss(features_styl, features_cont):
    """
    Given stylized features and content features from cross-attn modules,
    we penalize the difference. This is a simple L2 distance:
        L_att = || F_styl - A(F_cont) ||^2
    
    For demonstration, we’ll treat features_styl as the final styl features 
    and features_cont as the cross-attn-aligned representation. We assume
    the code that calls this function has already computed A(F_cont).
    """
    # If we want a simple alignment, we do a direct L2
    return F.mse_loss(features_styl, features_cont)

################################################################################
# 4. Original Content Loss
################################################################################
def content_loss_vgg(styl_features, content_features):
    """
    L_con = ||Phi(I_styl) - Phi(I_cont)||^2 using VGG features.
    styl_features and content_features are dicts from get_features() in utils.py,
    typically grabbing relevant layers (e.g. 'conv4_2','conv5_2', etc.)
    """
    loss = 0.0
    # For example, compare conv4_2 and conv5_2
    for layer in ['conv4_2','conv5_2']:
        loss += torch.mean((styl_features[layer] - content_features[layer])**2)
    return loss
