from argparse import Namespace
import os

# Example placeholder for style names
try:
    with open('wikiart_styles.txt', 'r') as f:
        style_names = [line.strip() for line in f.readlines()]
except:
    # Fallback if the file doesn't exist
    style_names = ["VanGogh", "Picasso", "Monet"]

# If you want some default subset for patch-based variations
lambda_patch_styles = style_names[:10]

##############################################################################
# Hyperparameters for AttnStyle
##############################################################################
args = Namespace(
    # Main weights for the new methodology
    lambda_cls=1.0,  # Global classification loss
    lambda_ms=1.0,   # Multi-scale classification loss
    lambda_att=1.0,  # Attention-based alignment loss
    lambda_con=1.0,  # Original content loss

    # Image processing
    img_height=512,
    img_width=512,
    img_size=512,  # Typically used inside the diffusion model

    # Training
    max_step=200,
    lr=5e-4,

    # Single/few-step diffusion parameters
    batch_size=1,
    dtype='float32',
    loss='l2',
    version='2-0',  # for stable diffusion v2.0
    diff_times=4,   # number of random timesteps to sample each iteration

    # Content path & style text
    content_path=os.path.join(os.getcwd(), "CLIPstyler/test_set/face.jpg"),
    text="A mystical forest scene",  # default text prompt

    # Additional
    lambda_patch_styles=lambda_patch_styles,
    style_names=style_names,
)

# You can add more fields if needed by your code
# e.g., args.device = 'cuda'
