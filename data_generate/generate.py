import os
import requests
import argparse


def call_o3_api(prompt: str, output_path: str, api_key: str, endpoint: str) -> None:
    """Call the O3 API with a text prompt and save the resulting file.

    This function assumes the API returns binary image data.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(endpoint, json={"prompt": prompt}, headers=headers)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)


def generate_pair(image_path: str, output_dir: str, description_prompt: str, generation_prompt_template: str,
                   api_key: str, endpoint: str) -> None:
    """Generate a description and a photorealistic image from an artwork."""
    basename = os.path.splitext(os.path.basename(image_path))[0]
    desc_prompt = description_prompt.format(image_path=image_path)
    # First call: get textual description
    desc_output = os.path.join(output_dir, f"{basename}_description.txt")
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.post(endpoint, json={"prompt": desc_prompt}, headers=headers)
    resp.raise_for_status()
    description = resp.json().get("text", "")
    with open(desc_output, "w") as f:
        f.write(description)

    # Second call: generate photo-realistic image
    final_prompt = generation_prompt_template.format(description=description)
    img_output = os.path.join(output_dir, f"{basename}_photo.png")
    call_o3_api(final_prompt, img_output, api_key, endpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate content-style pairs using the O3 API")
    parser.add_argument("--input-dir", required=True, help="Directory of source artwork images")
    parser.add_argument("--output-dir", required=True, help="Directory to write generated files")
    parser.add_argument("--endpoint", default=os.environ.get("O3_ENDPOINT", "https://api.o3.example/v1/generate"),
                        help="O3 API endpoint")
    parser.add_argument("--description-prompt", default="Describe the layout and objects in {image_path}, ignoring artistic style.",
                        help="Prompt template for obtaining descriptions")
    parser.add_argument("--generation-prompt", default="Generate a photorealistic DSLR-style image based on: {description}",
                        help="Prompt template for creating images")
    args = parser.parse_args()

    api_key = os.environ.get("O3_API_KEY")
    if not api_key:
        raise RuntimeError("O3_API_KEY environment variable not set")

    os.makedirs(args.output_dir, exist_ok=True)
    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        generate_pair(os.path.join(args.input_dir, fname), args.output_dir, args.description_prompt,
                      args.generation_prompt, api_key, args.endpoint)
