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
import os
import sys
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset.vp_visualizer import VanishingPointVisualizer


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


def main(
    vanishing_point, model_name, guidance_scale, controlnet_conditioning_scale, seed
):
    # download an image
    # dataset = load_from_disk("/srv/datasets3/HoliCity/dataset_w_vpts")
    # conditioning_image = dataset[5]["conditioning"]

    vp_visualizer = VanishingPointVisualizer(
        image_size=(512, 512),
        angle_step=5,
    )
    conditioning_image = vp_visualizer.create_condition_image(vanishing_point)
    conditioning_image = Image.fromarray(conditioning_image)
    # load control net and stable diffusion v1-5
    controlnet = ControlNetModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    # controlnet = ControlNetModel.from_pretrained(
    #     "/home/okumura/lab/vanishing_point/src/model_out/checkpoint-480000/controlnet",
    #     torch_dtype=torch.float16,
    # )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed
    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()

    # 通常のStable Diffusion パイプラインを設定
    base_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
    )
    base_pipe.scheduler = UniPCMultistepScheduler.from_config(
        base_pipe.scheduler.config
    )
    base_pipe.enable_xformers_memory_efficient_attention()
    base_pipe.enable_model_cpu_offload()

    # 通常のStable Diffusionで画像生成
    generator = torch.manual_seed(seed)
    base_images = base_pipe(
        "modern buildings, high quality, photorealistic",
        num_inference_steps=50,
        generator=generator,
        num_images_per_prompt=5,
        guidance_scale=guidance_scale,
    ).images

    # ControlNetを使用した画像生成
    generator = torch.manual_seed(seed)
    controlnet_images = pipe(
        "buildings, high quality, photorealistic",
        num_inference_steps=50,
        generator=generator,
        image=conditioning_image,
        num_images_per_prompt=5,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images

    # 画像の保存
    conditioning_image.save("conditioning_image.png")
    for i, image in enumerate(base_images):
        image.save(f"base_output_{i}.png")
    for i, image in enumerate(controlnet_images):
        image.save(f"controlnet_output_{i}.png")

    # 消失点の線を重ねる
    for i, image in enumerate(controlnet_images):
        image_with_lines = overlay_vanishing_point(image, vanishing_point)
        image_with_lines.save(f"controlnet_output_{i}_with_lines.png")


if __name__ == "__main__":
    # model_name = "/home/okumura/lab/vanishing_point/src/model_out_w_additional_canny_mask_10000/checkpoint-80000/controlnet"
    model_name = "/home/okumura/lab/vanishing_point/src/model_out_w_additional_canny_mask_1000/checkpoint-60000/controlnet"
    vanishing_point = np.array([[900, 400]])
    guidance_scale = 7.0
    controlnet_conditioning_scale = 0.400
    seed = 1
    main(
        vanishing_point, model_name, guidance_scale, controlnet_conditioning_scale, seed
    )
