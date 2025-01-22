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
    target_image,
    condition_image,
    model_name,
    guidance_scale,
    controlnet_conditioning_scale,
    seed,
):
    # load control net and stable diffusion v1-5
    controlnet = ControlNetModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    # controlnet = ControlNetModel.from_pretrained(
    #     "/home/okumura/lab/vanishing_point/src/model_out/checkpoint-480000/controlnet",
    #     torch_dtype=torch.float16,
    # )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
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
        "buildings on both sides of the road, high quality, photorealistic",
        num_inference_steps=20,
        generator=generator,
        num_images_per_prompt=20,
        guidance_scale=guidance_scale,
    ).images

    # ControlNetを使用した画像生成
    generator = torch.manual_seed(seed)
    controlnet_images = pipe(
        "modern buildings, high quality, photorealistic",
        image=condition_image,
        num_inference_steps=20,
        generator=generator,
        num_images_per_prompt=5,
        # guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images
    # controlnet_images = pipe(
    #     "",
    #     guess_mode=True,
    #     num_inference_steps=50,
    #     generator=generator,
    #     image=condition_image,
    #     num_images_per_prompt=5,
    #     guidance_scale=1.0,
    #     # controlnet_conditioning_scale=controlnet_conditioning_scale,
    # ).images
    # 画像の保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{os.path.dirname(__file__)}/output/{timestamp}"
    condition_image.save(f"{output_dir}/conditioning_image.png")
    for i, image in enumerate(base_images):
        image.save(f"{output_dir}/raw_SD/{timestamp}_{i}.png")
    for i, image in enumerate(controlnet_images):
        image.save(f"{output_dir}/raw_controlnet/{timestamp}_{i}.png")
    target_image.save(f"{output_dir}/raw_controlnet/target_{timestamp}.png")


if __name__ == "__main__":
    idx = 11
    model_name = "/home/okumura/lab/vanishing_point/ckpt/model_out_w_vpts_edges_black-bg2/checkpoint-3500/controlnet"
    dataset_path = "/srv/datasets3/HoliCity/dataset_w_vpts_edges"
    dataset = load_from_disk(dataset_path)
    target_image = dataset[idx]["image"]
    # edgesから条件画像を作成
    edges = dataset[idx]["edge"]
    condition_image = np.zeros((512, 512, 3), dtype=np.uint8)
    if edges:  # エッジが存在する場合
        for vp_edges in edges:
            for i in range(0, len(vp_edges), 4):
                x1, y1 = int(vp_edges[i]), int(vp_edges[i + 1])
                x2, y2 = int(vp_edges[i + 2]), int(vp_edges[i + 3])
                cv2.line(
                    condition_image, (x1, y1), (x2, y2), (255, 255, 255), 1
                )  # 白線で描画
    condition_image = Image.fromarray(condition_image)
    guidance_scale = 7.5
    controlnet_conditioning_scale = 1.0
    seed = 4
    main(
        target_image,
        condition_image,
        model_name,
        guidance_scale,
        controlnet_conditioning_scale,
        seed,
    )
