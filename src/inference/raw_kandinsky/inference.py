# !pip install diffusers transformers accelerate torch
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
from datetime import datetime
import os


def main(
    prompt,
):
    # Kandinsky パイプラインを設定
    pipe = AutoPipelineForText2Image.from_pretrained(
        "kandinsky-community/kandinsky-3",
        variant="fp16",
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()

    # Kandinskyで画像生成
    images = []
    for _ in range(10):
        image = pipe(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=25,
            generator=torch.Generator("cpu").manual_seed(
                torch.randint(0, 2**32, (1,)).item()
            ),
        ).images[0]
        images.append(image)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{os.path.dirname(__file__)}/output"
    os.makedirs(output_dir, exist_ok=True)

    # 各画像を個別に保存
    for idx, img in enumerate(images):
        output_path = f"{output_dir}/{timestamp}_{idx}.png"
        img.save(output_path)
        print(f"output image saved to {output_path}")


if __name__ == "__main__":
    # prompt = "buildings on both sides of the road, high quality, photorealistic"
    # prompt = "Row of buildings alongside a straight road, high quality, photorealistic"
    prompt = "Buildings along one side of a straight road, high quality, photorealistic"
    prompt = (
        "Straight road in front, buildings along the road, high quality, photorealistic"
    )
    prompt = "Side view of a street, road running horizontally from left to right, buildings on the far side only, high quality, photorealistic"
    prompt = "Side view of a street, road running horizontally from left to right, high quality, photorealistic"
    prompt = "Sidewalk in foreground, asphalt road in middle ground running left to right, row of shops and buildings in background, eye level view, high quality, photorealistic"
    prompt = "road"
    prompt = "row of buildings, high quality, photorealistic"
    # prompt = "Room, high quality, photorealistic"
    for i in range(1):
        main(
            prompt,
        )
