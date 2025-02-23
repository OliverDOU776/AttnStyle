import os
from models import load_vgg_model
import utils
import torch
from PIL import Image
import datetime
from train import train_attnstyle

def test_attnstyle(args, device, VGG, content_image, content_features, style_net, text_prompt, model_args):
    """
    Trains the AttnStyle network on a single content image + text prompt,
    logs the losses, and saves intermediate images.
    """
    # Name your log file with a timestamp
    start_date = datetime.datetime.now().strftime("%Y%m%d")
    
    # Prepare result directory
    result_dir = utils.get_result_path(args, "AttnStyle")
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"{'*' * 30}\nTraining AttnStyle with prompt: {text_prompt}\n{'*' * 30}")

    # Train the network
    loss_logs = train_attnstyle(
        args, device,
        VGG,
        content_image,
        content_features,
        style_net,
        text_prompt,
        result_dir,
        model_args
    )

    # Save loss logs
    log_file_path = os.path.join(result_dir, f"{start_date}_loss_logs.txt")
    with open(log_file_path, 'w') as log_file:
        for key in loss_logs:
            log_file.write(f"{key}: {loss_logs[key]}\n")

    print("Training complete.")
