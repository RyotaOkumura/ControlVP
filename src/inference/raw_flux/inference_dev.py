# !pip install diffusers transformers accelerate torch
from diffusers import FluxPipeline
from huggingface_hub import login
import torch
from datetime import datetime
import os


def main(
    prompt,
):
    # FLUX Dev パイプラインを設定
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    # FLUX Devで画像生成
    images = pipe(
        prompt=prompt,
        num_images_per_prompt=10,
        height=512,
        width=512,
        guidance_scale=3.5,  # FLUX.1-devは3.5が推奨
        num_inference_steps=50,  # guidance-distilledモデルは50ステップ推奨
        generator=torch.Generator("cuda").manual_seed(
            torch.randint(0, 2**32, (1,)).item()
        ),
    ).images
    # images = []
    # for _ in range(10):
    #     image = pipe(
    #         prompt=prompt,
    #         height=512,
    #         width=512,
    #         guidance_scale=3.5,  # FLUX.1-devは3.5が推奨
    #         num_inference_steps=50,  # guidance-distilledモデルは50ステップ推奨
    #         generator=torch.Generator("cuda").manual_seed(
    #             torch.randint(0, 2**32, (1,)).item()
    #         ),
    #     ).images[0]
    #     images.append(image)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{os.path.dirname(__file__)}/output"
    os.makedirs(output_dir, exist_ok=True)

    # 各画像を個別に保存
    prefix = 6
    for idx, img in enumerate(images):
        output_path = f"{output_dir}/{timestamp}_{prefix}_{idx}.png"
        img.save(output_path)
        print(f"output image saved to {output_path}")


if __name__ == "__main__":
    login(token="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")

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
    prompt = "row of buildings"
    # prompt = "Room, high quality, photorealistic"
    for i in range(1):
        main(
            prompt,
        )
