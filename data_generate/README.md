# Data Generation with O3 API

This folder contains a simple script to create content-style pairs from WikiArt images.
The script assumes you have an API key for the O3 text-to-image service.

## Usage

```bash
export O3_API_KEY=your_api_key
# optional: export O3_ENDPOINT=https://api.o3.example/v1/generate
python generate.py --input-dir path/to/wikiart --output-dir path/to/output
```

Each source image produces two files in the output directory:

- `<name>_description.txt` – text describing the image content.
- `<name>_photo.png` – a photorealistic rendition generated from the description.

## Example Prompts

You can experiment with different prompts by using the `--description-prompt`
and `--generation-prompt` arguments. Here are a few examples:

1. **Minimal layout**
   - Description prompt:
     `"List the main objects and where they appear in {image_path}. Ignore style."`
   - Generation prompt:
     `"Create a DSLR-style photograph depicting: {description}"`

2. **Stage directions**
   - Description prompt:
     `"Write stage directions describing the scene in {image_path} without mentioning artistic style."`
   - Generation prompt:
     `"Render a realistic photo matching these stage directions: {description}"`

3. **Spatial summary**
   - Description prompt:
     `"Summarize the spatial arrangement of subjects in {image_path}, omitting artistic brushwork."`
   - Generation prompt:
     `"Produce a detailed, photorealistic shot based on this summary: {description}"`

4. **Object list**
   - Description prompt:
     `"Provide a concise list of objects in {image_path} with their relative positions."`
   - Generation prompt:
     `"Create a lifelike photograph from this list: {description}"`

These variations can help determine which wording best removes artistic style
while preserving the underlying scene.
