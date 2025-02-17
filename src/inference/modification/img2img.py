from diffusers import (
    ControlNetModel,
    UniPCMultistepScheduler,
    AutoPipelineForImage2Image,
)
import numpy as np
import torch
from PIL import Image
import os
import cv2
from datetime import datetime
from diffusers.utils import make_image_grid


def create_result_grid(image_rows, base_size=None):
    """結果をグリッド形式の画像として作成する

    Args:
        image_rows (list[list[PIL.Image]]): 画像の2次元配列。各行には異なる数の画像を含められる
        base_size (tuple[int, int], optional): 空白画像のサイズ。Noneの場合は最初の画像のサイズを使用

    Returns:
        PIL.Image: グリッド形式の結果画像
    """
    if not image_rows or not image_rows[0]:
        raise ValueError("画像配列が空です")

    # 基準サイズの決定
    if base_size is None:
        # 最初の画像のサイズを使用
        base_size = image_rows[0][0].size

    # 空白の画像を作成
    blank_image = Image.new("RGB", base_size, (255, 255, 255))

    # 最大列数を計算
    max_cols = max(len(row) for row in image_rows)

    # 各行を最大列数に合わせて空白画像で埋める
    padded_rows = []
    for row in image_rows:
        padded_row = row + [blank_image] * (max_cols - len(row))
        padded_rows.extend(padded_row)

    return make_image_grid(
        padded_rows,
        rows=len(image_rows),
        cols=max_cols,
    )


def make_condition_image(condition_npz_path, line_width=1):
    edges = np.load(condition_npz_path)["edges"]
    # edgesを二要素ずつ読み込む
    condition_image = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(0, len(edges), 4):
        x1, y1 = int(edges[i]), int(edges[i + 1])
        x2, y2 = int(edges[i + 2]), int(edges[i + 3])
        cv2.line(
            condition_image, (x1, y1), (x2, y2), (255, 255, 255), line_width
        )  # 白線で描画
    return condition_image


def main():
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_PATH,
        torch_dtype=torch.float16,
    )
    controlnet_pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_NAME,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )

    # speed up diffusion process with faster scheduler and memory optimization
    controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(
        controlnet_pipe.scheduler.config
    )
    # remove following line if xformers is not installed
    controlnet_pipe.enable_xformers_memory_efficient_attention()

    controlnet_pipe.enable_model_cpu_offload()

    # 入力画像の読み込み
    input_image = Image.open(INIT_IMAGE_PATH)
    conditioning_image = Image.fromarray(make_condition_image(CONDITION_NPZ_PATH))

    input_image.save("input_image.png")
    # 通常のStable Diffusionで画像生成
    generator = torch.manual_seed(SEED)
    # ControlNetを使用した画像生成
    generator = torch.manual_seed(SEED)
    controlnet_images = controlnet_pipe(
        prompt=PROMPT,
        num_inference_steps=50,
        generator=generator,
        control_image=conditioning_image,
        num_images_per_prompt=10,
        guidance_scale=GUIDANCE_SCALE,
        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
        image=input_image,
        strength=STRENGTH,
    ).images

    # 画像の保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{os.path.dirname(__file__)}/output/"
    conditioning_image.save("conditioning_image.png")
    condition_for_overlay = make_condition_image(CONDITION_NPZ_PATH, line_width=2)
    grid_image = create_result_grid(
        [
            [input_image, conditioning_image],
            controlnet_images,
            [
                Image.fromarray(
                    np.where(
                        np.array(condition_for_overlay)[:, :, 0:1] > 128,
                        np.array(np.array([255, 0, 0])),
                        np.array(control_net_image),
                    ).astype(np.uint8)
                )
                for control_net_image in controlnet_images
            ],
        ]
    )
    grid_image.save(f"{output_dir}/img2img/{timestamp}_str-{STRENGTH}.png")


if __name__ == "__main__":
    INIT_IMAGE_PATH = (
        "/home/okumura/lab/vanishing_point/data/data_20250121_184745_imag.jpg"
    )
    CONDITION_NPZ_PATH = (
        "/home/okumura/lab/vanishing_point/data/data_20250121_184745_edge.npz"
    )
    CONTROLNET_MODEL_PATH = "/home/okumura/lab/vanishing_point/ckpt/model_out_w_vpts_edges_black-bg2/checkpoint-3500/controlnet"
    MODEL_NAME = "stabilityai/stable-diffusion-2-1"
    PROMPT = "modern buildings, high quality, photorealistic"
    GUIDANCE_SCALE = 7.5
    CONTROLNET_CONDITIONING_SCALE = 1.0
    SEED = 10
    STRENGTH = 0.5
    main()
