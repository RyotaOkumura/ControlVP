# !pip install diffusers transformers accelerate torch
from diffusers import FluxPipeline
from huggingface_hub import login
import torch
from datetime import datetime
import os


def main(
    prompt,
):
    # Setup FLUX Dev pipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    # Generate images with FLUX Dev
    images = pipe(
        prompt=prompt,
        num_images_per_prompt=10,
        height=512,
        width=512,
        guidance_scale=3.5,
        num_inference_steps=50,
        generator=torch.Generator("cuda").manual_seed(
            torch.randint(0, 2**32, (1,)).item()
        ),
    ).images

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{os.path.dirname(__file__)}/output"
    os.makedirs(output_dir, exist_ok=True)

    # Save each image
    prefix = 6
    for idx, img in enumerate(images):
        output_path = f"{output_dir}/{timestamp}_{prefix}_{idx}.png"
        img.save(output_path)
        print(f"output image saved to {output_path}")


if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    prompt = "Buildings on both sides of the road, high quality, photorealistic"
    prompt = "Row of buildings alongside a straight road, high quality, photorealistic"
    prompt = "Row of buildings, high quality, photorealistic"
    # prompt = "Room, high quality, photorealistic"
    for i in range(1):
        main(
            prompt,
        )
