import utils
utils.setup_paths()

import torch
from models import load_vgg_model, initialize_diffusion_model
from utils import load_content_image, generate_contrastive_samples
from test import test_styler
from config import args
import clip
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
vae, tokenizer, text_encoder, unet, scheduler = initialize_diffusion_model(args, device)
VGG = load_vgg_model(device)
content_image = load_content_image(args.content_path, args.img_height, args.img_width, device)
content_features = utils.get_features(utils.img_normalize(content_image, device), VGG)
clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)
source = "a Photo"
template_source = utils.compose_text_with_templates(source)
tokens_source = clip.tokenize(template_source).to(device)
with torch.no_grad():
    text_source = clip_model.encode_text(tokens_source).mean(axis=0, keepdim=True)
    text_source = text_source / text_source.norm(dim=-1, keepdim=True)
    source_features = clip_model.encode_image(utils.clip_normalize(content_image, device))
    source_features = source_features / source_features.norm(dim=-1, keepdim=True)



print(f"Processing style: {args.text}")
# Generate class names for CLIP loss
args.class_names = generate_contrastive_samples(args.text, n=5)
print(f"Class names for CLIP loss: {args.class_names}")

# Update text features
template_text = utils.compose_text_with_templates(args.text)
tokens = clip.tokenize(template_text).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(tokens).mean(axis=0, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

print(f"Using lambda_patch: {args.lambda_patch}")

# Create a unique identifier for saving results
result_suffix = f"{args.text}_lambda_patch_{args.lambda_patch}"
args.result_suffix = result_suffix  # Pass this to test_styler or wherever needed

# Run the test styler
test_styler(args, device, VGG, content_image, content_features, clip_model, text_features, text_source, source_features)