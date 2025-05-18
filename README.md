# AttnStyle

This project explores automatic style transfer for artworks. To train and evaluate models, we first create a dataset that separates semantic content from artistic style. The steps below outline a simple procedure using the O3 API.

## Dataset Generation Overview

1. **Describe the painting**: For each WikiArt image, send a prompt asking for a description of the scene layout and objects while ignoring brushwork and color style.
2. **Generate a neutral photo**: Use the returned description as input to a text-to-image model. Request a photorealistic DSLR-style rendition of the scene. This serves as the "content" image without the painting's style.
3. **Store pairs**: Save the text description and generated photo alongside the original artwork. These pairs can train or evaluate style-transfer systems.

See `data_generate/README.md` for a script and sample prompts.
