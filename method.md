### 3. Methodology

#### 3.1 Overview

AttnStyle performs zero-shot style transfer by transforming a content image into a stylized output guided solely by a text prompt. Our approach leverages a lightweight U-Net–based encoder–decoder architecture augmented with cross-attention modules to align content and stylized feature representations. In addition to the original content loss from CLIPstyler—which preserves low- and mid-level content details—we incorporate multi-scale patch classification and attention-based alignment losses. These additional losses enforce stylistic consistency at various resolutions and help maintain the semantic structure of the content image.

Notably, AttnStyle is designed as an online training algorithm. Instead of pre-training the model offline, the system optimizes the model for each individual content image and text prompt during inference, typically converging in 100–200 epochs.

#### 3.2 Network Architecture

Our network is built upon a U-Net architecture:
- **Encoder:** Extracts hierarchical features from the input content image \(I_{\text{cont}}\).
- **Decoder:** Synthesizes the stylized image \(I_{\text{styl}}\) from the encoded representation.
- **Cross-Attention Modules:** Embedded in intermediate layers, these modules align decoder features (\(F_{\text{styl}}\)) with encoder features (\(F_{\text{cont}}\)) via an attention mechanism. Here, decoder features serve as queries while the content features act as keys and values, enabling the network to adaptively integrate structural details during stylization.

#### 3.3 Loss Functions

The overall training objective is a weighted sum of multiple loss components that together ensure both stylistic fidelity to the text prompt and preservation of the content image’s structure:

\[
\begin{aligned}
L_{\text{total}} =\; & \lambda_{\text{cls}}\, L_{\text{cls}}(I_{\text{styl}}, T) \quad \text{(global classification loss)} \\
& + \lambda_{\text{ms}}\, \sum_{s \in \{\text{global, intermediate, local}\}} \lambda^{(s)} L^{(s)}_{\text{cls}}(I_{\text{styl}}^{(s)}, T) \quad \text{(multi-scale patch classification loss)} \\
& + \lambda_{\text{att}}\, L_{\text{att}}(F_{\text{styl}}, F_{\text{cont}}) \quad \text{(attention-based alignment loss)} \\
& + \lambda_{\text{con}}\, L_{\text{con}}(I_{\text{styl}}, I_{\text{cont}}) \quad \text{(original content loss)} \\
& + \lambda_{\text{layout}}\, L_{\text{layout}}(I_{\text{styl}}, I_{\text{cont}}) \quad \text{(textual layout map loss)} \\
& + \lambda_{\text{tv}}\, L_{\text{tv}}(I_{\text{styl}}) \quad \text{(total variation regularization)}
\end{aligned}
\]

**Global & Multi-Scale Patch Classification Loss:**  
To ensure the stylized output adheres to the style described by the text prompt \(T\), we use a zero-shot classifier (e.g., diffusion-based or CLIP-based) that evaluates the entire image as well as patches extracted at multiple scales:
- **Global Classification Loss \(L_{\text{cls}}\):** Measures the classifier’s confidence on the full image.
- **Multi-Scale Patch Classification Loss:** Patches \(I_{\text{styl}}^{(s)}\) are extracted at global, intermediate, and local scales. The losses \(L^{(s)}_{\text{cls}}\) computed at each scale are weighted by \(\lambda^{(s)}\) and aggregated.

**Attention-Based Alignment Loss:**  
The attention-based alignment loss preserves the semantic structure by aligning the stylized features \(F_{\text{styl}}\) with the content features \(F_{\text{cont}}\). A cross-attention module computes an aligned representation \(A(F_{\text{cont}})\) based on \(F_{\text{styl}}\). The loss is defined as:

\[
L_{\text{att}} = \|F_{\text{styl}} - A(F_{\text{cont}})\|_2^2,
\]

which encourages the stylized features to retain the structural essence of the content image.

**Original Content Loss:**  
We retain the original content loss from CLIPstyler, defined as the mean-squared error between features extracted (e.g., via a VGG network) from \(I_{\text{styl}}\) and \(I_{\text{cont}}\):

\[
L_{\text{con}} = \|\phi(I_{\text{styl}}) - \phi(I_{\text{cont}})\|_2^2.
\]


#### 3.4 Online Training Strategy

AttnStyle employs an online, per-instance optimization strategy. When a user inputs a content image \(I_{\text{cont}}\) and a text prompt \(T\), the model is trained from scratch on that pair for approximately 100–200 epochs. This rapid, on-the-fly training allows the system to tailor the stylization specifically to the input image and desired style without relying on a pre-trained model.

During this online training:
- **Dynamic Loss Weighting:**  
  A curriculum strategy is applied where early epochs emphasize the attention-based alignment to secure the structural foundation. As training progresses, the influence of the global and multi-scale classification losses is gradually increased to refine stylistic details.
  
- **Data Augmentation:**  
  Geometric and photometric transformations are applied to generate augmented patches, reinforcing the robustness of the multi-scale patch classification loss.

- **Optimization:**  
  The total loss \(L_{\text{total}}\) is minimized using gradient descent, updating the network parameters for the given content and text prompt until convergence.

#### 3.5 Inference

At test time, a user provides a content image \(I_{\text{cont}}\) and text prompt \(T\). The AttnStyle network is then quickly trained online (within 100–200 epochs) to generate a stylized image \(I_{\text{styl}}\) that reflects the stylistic cues of \(T\) while faithfully preserving the content and structure of the original image. The online training approach ensures that the model is optimally adapted to each individual input, providing a personalized style transfer experience.