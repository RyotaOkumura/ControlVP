# !pip install diffusers transformers accelerate torch
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
from datetime import datetime
import os


def main(
    prompt,
):
    # Setup SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()

    # Generate images with SDXL
    images = []
    for _ in range(10):
        image = pipe(
            prompt=prompt,
            height=512,
            width=512,
            generator=torch.Generator("cpu").manual_seed(
                torch.randint(0, 2**32, (1,)).item()
            ),
        ).images[0]
        images.append(image)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{os.path.dirname(__file__)}/output"
    os.makedirs(output_dir, exist_ok=True)

    # Save images
    for idx, img in enumerate(images):
        output_path = f"{output_dir}/{timestamp}_{idx}.png"
        img.save(output_path)
        print(f"output image saved to {output_path}")


if __name__ == "__main__":
    prompt = "Buildings on both sides of the road, high quality, photorealistic"
    prompt = "Row of buildings alongside a straight road, high quality, photorealistic"
    prompt = "Row of buildings, high quality, photorealistic"
    # prompt = "Room, high quality, photorealistic"
    for i in range(1):
        main(
            prompt,
        )
