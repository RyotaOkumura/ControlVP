import torch
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
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
        try:
            x1, y1 = int(edges[i]), int(edges[i + 1])
            x2, y2 = int(edges[i + 2]), int(edges[i + 3])
            cv2.line(
                condition_image, (x1, y1), (x2, y2), (255, 255, 255), line_width
            )  # 白線で描画
        except IndexError:
            # インデックスエラーが発生した場合は無視する
            pass
    return condition_image


def main(
    init_image_path,
    condition_npz_path,
    mask_image_path=None,
    output_dir=None,
    base_filename=None,
    index=0,
):
    """
    Args:
        init_image_path (str): 初期画像のパス
        condition_npz_path (str): 条件画像のnpzファイルパス
        mask_image_path (str, optional): マスク画像のパス。Noneの場合は自動生成
        output_dir (str, optional): 出力ディレクトリ
        base_filename (str, optional): 出力ファイル名のベース部分
    """
    # load model
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_NAME,
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

    mask_image_pil = Image.fromarray(mask_image)
    blurred_mask = pipeline.mask_processor.blur(mask_image_pil, blur_factor=BLUR_FACTOR)

    images = pipeline(
        prompt=PROMPT,
        image=init_image,
        mask_image=blurred_mask,
        strength=STRENGTH,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        # num_inference_steps=20,
        # guidance_scale=7.5,
    ).images

    # 出力ディレクトリが存在しない場合は作成
    if output_dir is None:
        output_dir = f"{os.path.dirname(__file__)}/output/sd_inpainting"

    os.makedirs(output_dir, exist_ok=True)

    # 各画像の保存
    for i, image in enumerate(images):
        if base_filename:
            # 指定されたベースファイル名を使用
            output_path = f"{output_dir}/{base_filename}_{index+i}.jpg"
        else:
            # タイムスタンプを使用
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if mask_image_path is None:
                output_path = f"{output_dir}/{timestamp}_str-{STRENGTH}_blur-{BLUR_FACTOR}_kernel-{KERNEL_SIZE}_image-{index+i}.jpg"
            else:
                output_path = f"{output_dir}/{timestamp}_w-mask_str-{STRENGTH}_blur-{BLUR_FACTOR}_kernel-{KERNEL_SIZE}_image-{index+i}.jpg"
        image.save(output_path)
        print(f"output image saved to {output_path}")

    return images


def make_mask(condition_image, kernel_size=100):
    # condition_imageの白い線の周りだけマスクする
    mask_image = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_image[condition_image == 255] = 255

    # マスクを膨張させて線の周りをマスクする
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask_image, kernel, iterations=1)

    return dilated_mask


if __name__ == "__main__":
    # データセットディレクトリの設定
    DATASET_DIR = (
        "/home/okumura/lab/grad_thesis_vp/vanishing_point/dataset_for_inference_flux"
    )
    OUTPUT_DIR = "/home/okumura/lab/grad_thesis_vp/vanishing_point/src/evaluation/output/sd_small_mask_max_strength_inpainting_flux"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # データセット内のファイルを取得
    image_files = [f for f in os.listdir(DATASET_DIR) if f.endswith("_imag.jpg")]

    MODEL_NAME = "stabilityai/stable-diffusion-2-inpainting"

    # その他のパラメータ設定
    # small mask: kernel_size=30, blur_factor=20, strength=0.5
    # large mask: kernel_size=150, blur_factor=100, strength=0.5
    KERNEL_SIZE = 30
    BLUR_FACTOR = 20
    STRENGTH = 1.0
    NUM_IMAGES_PER_PROMPT = 10
    PROMPT = "building, high quality, photorealistic"

    # 各画像に対して処理を実行
    for image_file in image_files:
        # ファイル名のベース部分を取得（例：data_20250429_191015_-130_375）
        base_name = image_file.replace("_imag.jpg", "")

        # 対応するファイルパスを構築
        init_image_path = os.path.join(DATASET_DIR, f"{base_name}_imag.jpg")
        condition_npz_path = os.path.join(DATASET_DIR, f"{base_name}_edge.npz")
        mask_image_path = os.path.join(DATASET_DIR, f"{base_name}_mask.jpg")

        # ファイルの存在確認
        if not os.path.exists(condition_npz_path):
            print(f"Warning: Condition file not found: {condition_npz_path}")
            continue

        if not os.path.exists(mask_image_path):
            print(f"Warning: Mask file not found: {mask_image_path}")
            mask_image_path = None

        print(f"Processing {base_name}...")

        # すでに処理済みかチェック
        existing_outputs = [f for f in os.listdir(OUTPUT_DIR) if base_name in f]
        if len(existing_outputs) == 100:
            print(f"Skipping {base_name} as output files already exist")
            continue

        # 画像生成を実行
        for i in range(1):
            main(
                init_image_path=init_image_path,
                condition_npz_path=condition_npz_path,
                mask_image_path=mask_image_path,
                output_dir=OUTPUT_DIR,
                base_filename=base_name,
                index=10 * i,
            )
