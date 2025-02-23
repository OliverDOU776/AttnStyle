#!/usr/bin/env python3
"""
concat.py

A script to generate concatenated images from a structured directory of results.
Usage:
    python concat.py 
"""

import os
import logging
from PIL import Image
from config import args  

RESULTS_DIR = 'results'
CONTENT_PATH = args.content_path
LAMBDA_PATCH_STYLES = args.lambda_patch_styles
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'concatenated_images')
VERBOSE = True

def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')

def get_loss_types(results_dir):
    loss_types = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]
    return loss_types

def get_style_prompts(results_dir, loss_types):
    style_prompts = set()
    if not loss_types:
        return style_prompts
    loss_dir = os.path.join(results_dir, loss_types[0])
    for entry in os.listdir(loss_dir):
        full_path = os.path.join(loss_dir, entry)
        if os.path.isdir(full_path):
            style_prompts.add(entry)
    return style_prompts

def get_patches(loss_dir, style_prompt):
    style_prompt_dir = os.path.join(loss_dir, style_prompt)
    patches = [
        d for d in os.listdir(style_prompt_dir)
        if os.path.isdir(os.path.join(style_prompt_dir, d)) and d.startswith("patch_")
    ]
    return patches

def collect_epochs(loss_types, results_dir, style_prompt, patch=None):
    epoch_files = set()
    for loss in loss_types:
        path = os.path.join(results_dir, loss, style_prompt)
        if patch:
            path = os.path.join(path, patch)
        if not os.path.isdir(path):
            continue
        files = [
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and f.startswith("epoch_") and f.endswith(".png")
        ]
        epoch_files.update(files)
    return epoch_files

def concatenate_images(image_list):
    widths, heights = zip(*(img.size for img in image_list))
    total_width = sum(widths)
    max_height = max(heights)
    concatenated_image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in image_list:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width
    return concatenated_image

def process_style_prompt(loss_types, results_dir, concatenated_dir, content_image, style_prompt, is_patched):
    if is_patched:
        example_loss_dir = os.path.join(results_dir, loss_types[0], style_prompt)
        patches = get_patches(os.path.join(results_dir, loss_types[0]), style_prompt)
        if not patches:
            return
        for patch in patches:
            epoch_files = collect_epochs(loss_types, results_dir, style_prompt, patch)
            for epoch in sorted(epoch_files):
                images_to_concatenate = [content_image]
                missing_images = False
                for loss in loss_types:
                    image_path = os.path.join(results_dir, loss, style_prompt, patch, epoch)
                    if os.path.isfile(image_path):
                        try:
                            img = Image.open(image_path).convert('RGB')
                            images_to_concatenate.append(img)
                        except:
                            missing_images = True
                            break
                    else:
                        missing_images = True
                        break
                if missing_images:
                    continue
                concatenated_image = concatenate_images(images_to_concatenate)
                output_path = os.path.join(concatenated_dir, style_prompt, patch)
                os.makedirs(output_path, exist_ok=True)
                save_path = os.path.join(output_path, epoch)
                concatenated_image.save(save_path)
    else:
        epoch_files = collect_epochs(loss_types, results_dir, style_prompt)
        for epoch in sorted(epoch_files):
            images_to_concatenate = [content_image]
            missing_images = False
            for loss in loss_types:
                image_path = os.path.join(results_dir, loss, style_prompt, epoch)
                if os.path.isfile(image_path):
                    try:
                        img = Image.open(image_path).convert('RGB')
                        images_to_concatenate.append(img)
                    except:
                        missing_images = True
                        break
                else:
                    missing_images = True
                    break
            if missing_images:
                continue
            concatenated_image = concatenate_images(images_to_concatenate)
            output_path = os.path.join(concatenated_dir, style_prompt)
            os.makedirs(output_path, exist_ok=True)
            save_path = os.path.join(output_path, epoch)
            concatenated_image.save(save_path)

def main():
    setup_logging(VERBOSE)
    if not os.path.isfile(CONTENT_PATH):
        logging.error(f"Content image does not exist: {CONTENT_PATH}")
        return
    
    try:
        content_image = Image.open(CONTENT_PATH).convert('RGB')
    except:
        logging.error(f"Error loading content image {CONTENT_PATH}")
        return

    if not os.path.isdir(RESULTS_DIR):
        logging.error(f"Results directory does not exist: {RESULTS_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    loss_types = get_loss_types(RESULTS_DIR)
    if not loss_types:
        return

    style_prompts = get_style_prompts(RESULTS_DIR, loss_types)
    if not style_prompts:
        return

    for style_prompt in sorted(style_prompts):
        is_patched = style_prompt in LAMBDA_PATCH_STYLES
        process_style_prompt(
            loss_types=loss_types,
            results_dir=RESULTS_DIR,
            concatenated_dir=OUTPUT_DIR,
            content_image=content_image,
            style_prompt=style_prompt,
            is_patched=is_patched
        )

if __name__ == "__main__":
    main()
