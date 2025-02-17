import torch
import numpy as np
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers.utils import make_image_grid
from PIL import Image
import cv2
import os
from datetime import datetime


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
    """
    Args:
        controlnet_model_path (str): ControlNetのパス
        model_name (str): StableDiffusionのパス
        init_image (PIL.Image): 初期画像
        condition_image (numpy.ndarray): 条件画像、shape=(512, 512, 3)、dtype=uint8
        mask_image (numpy.ndarray): マスク画像、shape=(512, 512, 3)、dtype=uint8
        blur_factor (int, optional): マスクのぼかし係数。デフォルト値は0
    """
    # load models
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_PATH,
        torch_dtype=torch.float16,
    )
    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        MODEL_NAME,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipeline.enable_model_cpu_offload()
    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    pipeline.enable_xformers_memory_efficient_attention()

    # prepare images
    init_image = Image.open(INIT_IMAGE_PATH)
    condition_image = make_condition_image(CONDITION_NPZ_PATH)
    mask_image = make_mask(condition_image, kernel_size=KERNEL_SIZE)
    blurred_mask = pipeline.mask_processor.blur(
        Image.fromarray(mask_image), blur_factor=BLUR_FACTOR
    )
    # control_imageを正しい形状に変換（HWC -> CHW）
    control_image = condition_image.copy()  # numpy.ndarray (H, W, C)
    control_image = Image.fromarray(control_image)  # PIL.Image に変換

    images = pipeline(
        prompt=PROMPT,
        image=init_image,
        mask_image=blurred_mask,
        control_image=control_image,  # PIL.Image として渡す
        strength=STRENGTH,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        # num_inference_steps=20,
        # guidance_scale=7.5,
        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
    ).images
    condition_for_overlay = make_condition_image(CONDITION_NPZ_PATH, line_width=2)
    print(pipeline.unet.config.in_channels)
    # グリッド画像の作成
    grid_image = create_result_grid(
        [
            [init_image, control_image, blurred_mask],
            images,
            [
                Image.fromarray(
                    np.where(
                        np.array(condition_for_overlay)[:, :, 0:1] > 128,
                        np.array(np.array([255, 0, 0])),
                        np.array(image),
                    ).astype(np.uint8)
                )
                for image in images
            ],
        ]  # 1行目  # 2行目
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{os.path.dirname(__file__)}/output/inpainting/{timestamp}_str-{STRENGTH}_blur-{BLUR_FACTOR}_kernel-{KERNEL_SIZE}.jpg"
    grid_image.save(output_path)
    print(f"output image saved to {output_path}")

    if SAVE_EACH_IMAGE:
        i = 0
        for image in images:
            output_path = f"{os.path.dirname(__file__)}/output/inpainting/{timestamp}_str-{STRENGTH}_blur-{BLUR_FACTOR}_kernel-{KERNEL_SIZE}_image-{i}.jpg"
            image.save(output_path)
            print(f"output image saved to {output_path}")
            i += 1


def make_mask(condition_image, kernel_size=100):
    # condition_imageの白い線の周りだけマスクする
    mask_image = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_image[condition_image == 255] = 255

    # マスクを膨張させて線の周りをマスクする
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask_image, kernel, iterations=1)

    return dilated_mask


if __name__ == "__main__":
    CONTROLNET_MODEL_PATH = "/home/okumura/lab/vanishing_point/ckpt/model_out_w_vpts_edges_black-bg2/checkpoint-3500/controlnet"
    MODEL_NAME = "stabilityai/stable-diffusion-2-1"
    # INIT_IMAGE_PATH = (
    #     "/home/okumura/lab/vanishing_point/data/data_20250130_182025_imag.jpg"
    # )
    # # INIT_IMAGE_PATH = "/home/okumura/lab/vanishing_point/src/inference/modification/output/inpainting/20250122_190110_str-0.45_blur-10_kernel-100_image-1.jpg"
    # CONDITION_NPZ_PATH = (
    #     "/home/okumura/lab/vanishing_point/data/data_20250130_182025_edge.npz"
    # )
    INIT_IMAGE_PATH = (
        "/home/okumura/lab/vanishing_point/data/data_20250130_212334_imag.jpg"
    )
    CONDITION_NPZ_PATH = (
        "/home/okumura/lab/vanishing_point/data/data_20250130_212334_edge.npz"
    )
    # INIT_IMAGE_PATH = (
    #     "/home/okumura/lab/vanishing_point/data/data_20250130_194727_imag.jpg"
    # )
    # CONDITION_NPZ_PATH = (
    #     "/home/okumura/lab/vanishing_point/data/data_20250130_194727_edge.npz"
    # )
    KERNEL_SIZE = 150
    STRENGTH = 0.5
    BLUR_FACTOR = 100
    NUM_IMAGES_PER_PROMPT = 10
    PROMPT = "building, high quality, photorealistic"
    SAVE_EACH_IMAGE = True
    CONTROLNET_CONDITIONING_SCALE = 1.0
    for i in range(1):
        main()
