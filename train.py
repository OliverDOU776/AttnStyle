import torch
import torch.optim as optim
from torchvision import transforms
import utils
import time
import os
from tqdm import tqdm
from PIL import Image

# Import new losses
from Losses import (
    classification_loss,
    multi_scale_classification_loss,
    attention_alignment_loss,
    content_loss_vgg
)

def train_attnstyle(
    args, device,
    VGG,                  # For content loss
    content_image,        # (B=1,3,H,W)
    content_features,     # Precomputed VGG features for content
    style_net,            # AttnStyleUNet
    text_prompt,          # The style text prompt
    result_dir,           # Where to save intermediate outputs
    model_args            # Contains vae, tokenizer, text_encoder, unet, scheduler, etc.
):
    """
    Performs the on-the-fly style transfer training (100–200 epochs).
    The total loss:
        L_total = λ_cls * L_cls (global classification) +
                  λ_ms  * L_ms  (multi-scale classification) +
                  λ_att * L_att (attention-based alignment) +
                  λ_con * L_con (content loss).
    """
    os.makedirs(result_dir, exist_ok=True)

    # Optimizer
    optimizer = optim.Adam(style_net.parameters(), lr=args.lr)
    steps = args.max_step

    # Track losses
    loss_logs = {
        'total_loss': [],
        'global_cls_loss': [],
        'ms_cls_loss': [],
        'att_loss': [],
        'content_loss': []
    }

    # If your style_net needs the content encoder pass in advance:
    # We do one forward pass on content to get the "content_feats".
    with torch.no_grad():
        c1, c2, c3 = style_net.encode(content_image)
        content_feats_tuple = (c1, c2, c3)

    # Training loop
    pbar = tqdm(range(steps + 1), desc=f"Training AttnStyle for '{text_prompt}'")
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Forward pass
        stylized_out, styl_feats_enc, _ = style_net(content_image, content_feats_tuple)
        # stylized_out is the final stylized image
        # styl_feats_enc is (e1,e2,e3) for stylized, but we also have content_feats_tuple for the alignment

        # Compute Losses ------------------------------------------
        
        # 1) Global classification
        cls_loss_global = classification_loss(stylized_out, text_prompt, model_args)

        # 2) Multi-scale classification
        cls_loss_ms = multi_scale_classification_loss(stylized_out, text_prompt, model_args)

        # 3) Attention alignment
        # For demonstration, let’s compare the highest-level features
        # from styl_feats_enc[-1] and content_feats_tuple[-1], flattened
        styl_high = styl_feats_enc[-1]  # (B,256,H/8,W/8)
        cont_high = content_feats_tuple[-1]
        att_loss_val = attention_alignment_loss(styl_high, cont_high)

        # 4) Content loss (VGG)
        stylized_norm = utils.img_normalize(stylized_out, device)
        styl_vgg_feats = utils.get_features(stylized_norm, VGG)
        cont_loss_val = content_loss_vgg(styl_vgg_feats, content_features)

        # Weighted sum
        total_loss = (
            args.lambda_cls * cls_loss_global +
            args.lambda_ms  * cls_loss_ms +
            args.lambda_att * att_loss_val +
            args.lambda_con * cont_loss_val
        )

        # Backprop + step
        total_loss.backward()
        optimizer.step()

        # Logging
        loss_logs['total_loss'].append(total_loss.item())
        loss_logs['global_cls_loss'].append(cls_loss_global.item())
        loss_logs['ms_cls_loss'].append(cls_loss_ms.item())
        loss_logs['att_loss'].append(att_loss_val.item())
        loss_logs['content_loss'].append(cont_loss_val.item())

        # Save occasional images
        if epoch % 20 == 0 or epoch == steps:
            with torch.no_grad():
                out_np = utils.im_convert2(stylized_out.detach().cpu())
                out_np = (out_np * 255).astype('uint8')
                pil_out = Image.fromarray(out_np)
                save_path = os.path.join(result_dir, f"epoch_{epoch}.png")
                pil_out.save(save_path)

    return loss_logs
