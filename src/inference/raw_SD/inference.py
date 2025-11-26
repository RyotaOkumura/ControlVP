# !pip install opencv-python transformers accelerate
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from datasets import load_from_disk
import numpy as np
import torch
from PIL import Image
import cv2
from datetime import datetime
import os


def main(
    prompt,
):
    # 通常のStable Diffusion パイプラインを設定
    base_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    base_pipe.scheduler = UniPCMultistepScheduler.from_config(
        base_pipe.scheduler.config
    )
    base_pipe.enable_xformers_memory_efficient_attention()
    base_pipe.enable_model_cpu_offload()

    # 通常のStable Diffusionで画像生成
    base_images = base_pipe(
        prompt,
        num_images_per_prompt=10,
    ).images

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{os.path.dirname(__file__)}/output"
    os.makedirs(output_dir, exist_ok=True)

    # 各画像を個別に保存
    for idx, img in enumerate(base_images):
        output_path = f"{output_dir}/{timestamp}_{idx}.png"
        img.save(output_path)
        print(f"output image saved to {output_path}")


if __name__ == "__main__":
    # prompt = "buildings on both sides of the road, high quality, photorealistic"
    # prompt = "Row of buildings alongside a straight road, high quality, photorealistic"
    prompt = "Room, high quality, photorealistic"
    for i in range(10):
        main(
            prompt,
        )
