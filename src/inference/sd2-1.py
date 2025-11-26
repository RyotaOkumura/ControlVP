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
import random


def overlay_vanishing_point(image, vanishing_points):
    """
    消失点から放射状に直線を引いてimageに重ねる
    image: PIL Image
    vanishing_points: np.array shape=(N, 2) Nは消失点の数
    """
    # 画像をnumpy配列に変換
    img_array = np.array(image)

    # 画像サイズを取得
    h, w = img_array.shape[:2]

    # 角度のステップ（10度ごと）
    angle_step = 5
    angles = np.arange(0, 360, angle_step)

    # 各消失点について処理
    for vp in vanishing_points:
        # 各角度について直線を描画
        for angle in angles:
            theta = np.deg2rad(angle)
            direction = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)

            t = np.linspace(0, max(h, w) * 5, 100)
            points = vp.reshape(2, 1) + direction * t

            # 画像内の点のみをフィルタリング
            mask = (
                (points[0] >= 0) & (points[0] < w) & (points[1] >= 0) & (points[1] < h)
            )
            points = points[:, mask]

            # 直線を描画
            if points.shape[1] >= 2:
                for j in range(points.shape[1] - 1):
                    p1 = tuple(map(int, points[:, j]))
                    p2 = tuple(map(int, points[:, j + 1]))
                    cv2.line(img_array, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), 1)

    # numpy配列をPIL Imageに戻す
    image_with_lines = Image.fromarray(img_array)
    return image_with_lines


def main(guidance_scale, seed, num_images_per_prompt, prompt, save_all=False):
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
    generator = torch.manual_seed(seed)
    base_images = base_pipe(
        prompt,
        num_inference_steps=20,
        generator=generator,
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=guidance_scale,
    ).images

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{os.path.dirname(__file__)}/output"
    os.makedirs(output_dir, exist_ok=True)

    # グリッド用の画像リストを作成
    num_images = len(base_images)
    grid_images = []

    # 2行目以降: base_images
    grid_images.extend(base_images)

    # グリッドの作成
    rows = num_images  # target画像の行 + 生成画像の行
    cols = 1
    cell_size = 512  # 画像サイズ

    # 空の画像を作成
    grid = Image.new("RGB", (cell_size * cols, cell_size * rows))

    # グリッドに画像を配置
    for idx, img in enumerate(grid_images):
        grid.paste(img, (0, idx * cell_size))

    # グリッド画像を保存
    grid.save(f"{output_dir}/{timestamp}_comparison.png")
    print(f"output image saved to {output_dir}/{timestamp}_comparison.png")

    if save_all:
        for i, image in enumerate(base_images):
            image.save(f"{output_dir}/{timestamp}_{i}.png")


if __name__ == "__main__":
    guidance_scale = 7.5
    seed = random.randint(0, 1000000)
    print(f"seed: {seed}")
    # seed = 507132
    num_images_per_prompt = 30
    # prompt = "lots of houses and apartments and buildings, city, seen from ground, high quality, photorealistic"
    prompt = "apartments, houses, buildings, road, seen from ground, high quality, photorealistic"

    main(guidance_scale, seed, num_images_per_prompt, prompt, save_all=True)
