# !pip install diffusers transformers accelerate torch
from diffusers import PixArtAlphaPipeline
import torch
from datetime import datetime
import os
import random


def main(
    prompts,
):
    # Setup PixArt pipeline
    pipe = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()

    # Generate images with PixArt
    # Randomly select one prompt and generate 10 images
    selected_prompt = random.choice(prompts)

    images = pipe(
        prompt=selected_prompt,
        num_images_per_prompt=10,
        height=512,
        width=512,
        generator=torch.Generator("cpu").manual_seed(
            torch.randint(0, 2**32, (1,)).item()
        ),
    ).images

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{os.path.dirname(__file__)}/output"
    os.makedirs(output_dir, exist_ok=True)

    # Save each image
    prefix = 7
    for idx, img in enumerate(images):
        output_path = f"{output_dir}/{timestamp}_{prefix}_{idx}.png"
        img.save(output_path)
        print(f"output image saved to {output_path}")


if __name__ == "__main__":
    prompts = [
        "Buildings on both sides of the road, high quality, photorealistic"
        "Row of buildings alongside a straight road, high quality, photorealistic",
        "Row of buildings, high quality, photorealistic",
    ]

    for i in range(200):
        main(
            prompts,
        )
