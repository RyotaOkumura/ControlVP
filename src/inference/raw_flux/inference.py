# !pip install diffusers transformers accelerate torch
from diffusers import FluxPipeline
import torch
from PIL import Image
from datetime import datetime
import os


def main(
    prompt,
):
    # FLUX パイプラインを設定
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    # FLUXで画像生成
    images = []
    for _ in range(10):
        image = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=0.0,  # FLUX.1-schnellはguidance_scale=0.0を使用
            num_inference_steps=4,  # schnellモデルは少ないステップ数で動作
            max_sequence_length=256,  # schnellモデルに推奨
            generator=torch.Generator("cuda").manual_seed(torch.randint(0, 2**32, (1,)).item())
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
    prompt = "Room, high quality, photorealistic"
    for i in range(10):
        main(
            prompt,
        )