import utils
utils.setup_paths()

import torch
from models import load_vgg_model, initialize_diffusion_model, AttnStyleUNet
from utils import load_content_image, get_features
from config import args
from test import test_attnstyle

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the single/few-step diffusion model (DMD2 or Stable Diff)
    vae, tokenizer, text_encoder, unet, scheduler = initialize_diffusion_model(args, device)

    # Load VGG for content loss
    VGG = load_vgg_model(device)

    # Load content image
    content_image = load_content_image(args.content_path, args.img_height, args.img_width, device)

    # Precompute content features
    content_norm = utils.img_normalize(content_image, device)
    content_features = get_features(content_norm, VGG)

    # Prepare the model_args with references to the diffusion components
    model_args = args
    model_args.vae = vae
    model_args.tokenizer = tokenizer
    model_args.text_encoder = text_encoder
    model_args.unet = unet
    model_args.scheduler = scheduler
    model_args.device = device

    # Create the AttnStyle network
    style_net = AttnStyleUNet().to(device)

    # Run test (which internally calls train_attnstyle)
    test_attnstyle(
        args=args,
        device=device,
        VGG=VGG,
        content_image=content_image,
        content_features=content_features,
        style_net=style_net,
        text_prompt=args.text,
        model_args=model_args
    )
