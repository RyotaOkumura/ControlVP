import torch
import numpy as np
from diffusers import ControlNetModel
from diffusers.utils import make_image_grid
from PIL import Image
import cv2
import os
import sys
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.pipelines import StableDiffusionControlNetInpaintCFGPipeline


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


def main(init_image_path, condition_npz_path, mask_image_path=None):
    """
    Args:
        init_image_path (str): 初期画像のパス
        condition_npz_path (str): 条件画像のnpzファイルパス
        mask_image_path (str, optional): マスク画像のパス。Noneの場合は自動生成
    """
    # load models
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_PATHS[CONTROLNET_MODEL_IDX],
        torch_dtype=torch.float16,
    )
    pipeline = StableDiffusionControlNetInpaintCFGPipeline.from_pretrained(
        MODEL_NAME,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipeline.enable_model_cpu_offload()
    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    pipeline.enable_xformers_memory_efficient_attention()

    # prepare images
    init_image = Image.open(init_image_path)
    condition_image = make_condition_image(condition_npz_path)

    # マスク画像の準備
    if mask_image_path is None:
        mask_image = make_mask(condition_image, kernel_size=KERNEL_SIZE)
    else:
        mask_image = np.array(Image.open(mask_image_path))
        kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
        mask_image = cv2.dilate(mask_image, kernel, iterations=1)

    blurred_mask = pipeline.mask_processor.blur(
        Image.fromarray(mask_image), blur_factor=BLUR_FACTOR
    )
    control_image = Image.fromarray(condition_image)

    images = pipeline(
        prompt=PROMPT,
        image=init_image,
        mask_image=blurred_mask,
        control_image=control_image,  # PIL.Image として渡す
        strength=STRENGTH,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        controlnet_guidance_scale=CONTROLNET_GUIDANCE_SCALE,
        # num_inference_steps=20,
        # guidance_scale=7.5,
        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
    ).images
    condition_for_overlay = make_condition_image(condition_npz_path, line_width=2)
    print(pipeline.unet.config.in_channels)

    # 出力パスの設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ""
    if "w_tog_loss" in CONTROLNET_MODEL_PATHS[CONTROLNET_MODEL_IDX]:
        output_dir = f"{os.path.dirname(__file__)}/output/inpainting/w_tog_loss"
    elif "contour_vp_loss" in CONTROLNET_MODEL_PATHS[CONTROLNET_MODEL_IDX]:
        output_dir = f"{os.path.dirname(__file__)}/output/inpainting/w_vp_loss"
    elif "wo_vp-loss" in CONTROLNET_MODEL_PATHS[CONTROLNET_MODEL_IDX]:
        output_dir = f"{os.path.dirname(__file__)}/output/inpainting/wo_vp_loss"

    else:
        raise ValueError("Invalid controlnet model path")
    output_path = ""
    if mask_image_path is None:
        output_path = f"{output_dir}/{timestamp}_str-{STRENGTH}_blur-{BLUR_FACTOR}_kernel-{KERNEL_SIZE}.jpg"
    else:
        output_path = f"{output_dir}/{timestamp}_w-mask_str-{STRENGTH}_blur-{BLUR_FACTOR}_kernel-{KERNEL_SIZE}.jpg"

    # グリッド画像の保存
    if SAVE_GRID_IMAGE:
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

        grid_image.save(output_path)
        print(f"output image saved to {output_path}")

    # 各画像の保存
    if SAVE_EACH_IMAGE:
        i = 0
        for image in images:
            if mask_image_path is None:
                output_path = f"{output_dir}/{timestamp}_str-{STRENGTH}_blur-{BLUR_FACTOR}_kernel-{KERNEL_SIZE}_image-{i}.jpg"
            else:
                output_path = f"{output_dir}/{timestamp}_w-mask_str-{STRENGTH}_blur-{BLUR_FACTOR}_kernel-{KERNEL_SIZE}_image-{i}.jpg"
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
    IMAGE_PATHS = [
        {
            "init": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data/data_20250328_192600_imag.jpg",
            "condition": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data/data_20250328_192600_edge.npz",
            # "mask": None,
            "mask": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data/data_20250328_192600_mask.jpg",
        },
        {
            "init": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data/data_20250324_054933_imag.jpg",
            "condition": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data/data_20250324_054933_edge.npz",
            # "mask": None,
            "mask": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data/data_20250324_054933_mask.jpg",
        },
        {
            "init": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data/data_20250324_061130_imag.jpg",
            "condition": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data/data_20250324_061130_edge.npz",
            # "mask": None,
            "mask": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data/data_20250324_061130_mask.jpg",
        },
        {
            "init": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data/data_20250521_043108_imag.jpg",
            "condition": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data/data_20250521_043108_edge.npz",
            # "mask": None,
            "mask": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data/data_20250521_043108_mask.jpg",
        },
        {
            "init": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250523_194041_277_312_imag.jpg",
            "condition": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250523_194041_277_312_edge.npz",
            # "mask": None,
            "mask": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250523_194041_277_312_mask.jpg",
        },
        {
            "init": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250523_205525_296_235_imag.jpg",
            "condition": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250523_205525_296_235_edge.npz",
            # "mask": None,
            "mask": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250523_205525_296_235_mask.jpg",
        },
        {
            "init": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250523_232135_164_237_imag.jpg",
            "condition": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250523_232135_164_237_edge.npz",
            # "mask": None,
            "mask": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250523_232135_164_237_mask.jpg",
        },
        {
            "init": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250523_235821_242_247_imag.jpg",
            "condition": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250523_235821_242_247_edge.npz",
            # "mask": None,
            "mask": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250523_235821_242_247_mask.jpg",
        },
        {
            "init": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250524_001444_319_266_imag.jpg",
            "condition": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250524_001444_319_266_edge.npz",
            # "mask": None,
            "mask": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250524_001444_319_266_mask.jpg",
        },
        {
            "init": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250503_162346_256_232_imag.jpg",
            "condition": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250503_162346_256_232_edge.npz",
            # "mask": None,
            "mask": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250503_162346_256_232_mask.jpg",
        },
        {
            "init": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250504_134743_242_390_imag.jpg",
            "condition": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250504_134743_242_390_edge.npz",
            # "mask": None,
            "mask": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250504_134743_242_390_mask.jpg",
        },
        {
            "init": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250719_122634_430_-1_imag.jpg",
            "condition": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250719_122634_430_-1_edge.npz",
            # "mask": None,
            "mask": "/home/okumura/lab/grad_thesis_vp/vanishing_point/data_20250719_122634_430_-1_mask.jpg",
        },
    ]

    CONTROLNET_MODEL_PATHS = [
        "/home/okumura/lab/vanishing_point/ckpt/contour/successful/contour_best_wo_vp-loss/checkpoint-3500/controlnet",
        "/home/okumura/lab/grad_thesis_vp/vanishing_point/ckpt/contour/successful/model_out_contour_vp_loss_w-1000_v-pred/checkpoint-25500/controlnet",
        "/home/okumura/lab/grad_thesis_vp/vanishing_point/src/train/model_out_contour_vp_loss_w_tog_loss/checkpoint-12500/controlnet",
        "/home/okumura/lab/grad_thesis_vp/vanishing_point/src/train/model_out_contour_vp_loss_sd2-base/checkpoint-25000/controlnet",
    ]
    CONTROLNET_MODEL_IDX = 1
    MODEL_NAME = "stabilityai/stable-diffusion-2-inpainting"

    # その他のパラメータ設定
    # 元々: 30, 20
    KERNEL_SIZE = 30
    BLUR_FACTOR = 20
    STRENGTH = 1.0
    NUM_IMAGES_PER_PROMPT = 10
    PROMPT = "building, high quality, photorealistic"
    SAVE_EACH_IMAGE = True
    SAVE_GRID_IMAGE = True
    CONTROLNET_CONDITIONING_SCALE = 1.0
    CONTROLNET_GUIDANCE_SCALE = 3.0

    # インデックス0の画像セットで実行
    image_set_indexes = [11]
    paths = [IMAGE_PATHS[image_set_index] for image_set_index in image_set_indexes]

    for i in range(len(image_set_indexes)):
        for _ in range(1):
            main(paths[i]["init"], paths[i]["condition"], paths[i]["mask"])
