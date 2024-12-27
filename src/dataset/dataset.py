from datasets import Dataset
import os
from PIL import Image
import numpy as np
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.dataset.vp_visualizer import VanishingPointVisualizer


def create_dataset_from_images(image_base_dir, vpts_base_dir):
    """
    画像フォルダとvanishing pointsフォルダからデータセットを作成する

    Args:
        image_base_dir (str): 画像が含まれるベースディレクトリ
        vpts_base_dir (str): 消失点データが含まれるベースディレクトリ
    """
    image_paths = []
    vpts_paths = []

    # サブディレクトリを再帰的に探索
    for root, _, files in os.walk(image_base_dir):
        for file in files:
            if file.endswith(("jpg", "jpeg", "png")):
                # 画像パスを取得
                img_path = os.path.join(root, file)

                # 対応する消失点ファイルのパスを構築
                relative_path = os.path.relpath(root, image_base_dir)
                vpts_file = file.replace(
                    "_imag.jpg", "_vpts.npz"
                )  # ファイル名のパターンを変換
                vpts_path = os.path.join(vpts_base_dir, relative_path, vpts_file)

                # 消失点ファイルが存在する場合のみデータセットに追加
                if os.path.exists(vpts_path):
                    image_paths.append(img_path)
                    vpts_paths.append(vpts_path)

    # Visualizerを初期化
    visualizer = VanishingPointVisualizer(
        image_size=(512, 512),
        angle_step=10,
    )
    # 画像とconditioningを読み込む
    images = [Image.open(path) for path in image_paths]
    conditioning_images = [
        make_conditioning_image(path, visualizer) for path in vpts_paths
    ]
    captions = [""] * len(images)
    vanishing_points = [return_valid_vpts(path) for path in vpts_paths]

    dataset = Dataset.from_dict(
        {
            "image": images,
            "conditioning": conditioning_images,
            "caption": captions,
            "vanishing_points": vanishing_points,
        }
    )

    return dataset


def return_valid_vpts(vpts_path):
    """
    消失点データから条件画像を作成する
    """
    vpts_data = np.load(vpts_path)
    vpts_3d = vpts_data["vpts"]
    vpts_confidences = vpts_data["confidence"]
    FOCAL_LENGTH = 1
    IMAGE_SIZE = 512

    def vpt3d_to_2d(w):
        x = w[0] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
        y = -w[1] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
        return x, y

    valid_vpts = [
        vpt3d_to_2d(vpts_3d[i])
        for i in range(min(3, len(vpts_3d)))
        if vpts_confidences[i] > 500
    ]
    valid_vpts = valid_vpts[:3]

    return valid_vpts


def make_conditioning_image(vpts_path, visualizer):
    """
    消失点データから条件画像を作成する
    """
    valid_vpts = return_valid_vpts(vpts_path)
    # 消失点がない場合は白画像を返す
    if not valid_vpts:
        white_image = Image.new("RGB", (512, 512), "white")
        return white_image

    # 条件画像を生成
    condition_image = visualizer.create_condition_image(valid_vpts)

    # NumPy配列からPIL Imageに変換
    return Image.fromarray(condition_image)


if __name__ == "__main__":
    from src.config import config

    IMAGE_BASE_DIR = config["dataset"]["image_base_dir"]
    VPTS_BASE_DIR = config["dataset"]["vpts_base_dir"]
    DATASET_DIR = config["dataset"]["dataset_dir"]
    os.makedirs(DATASET_DIR, exist_ok=True)
    dataset = create_dataset_from_images(IMAGE_BASE_DIR, VPTS_BASE_DIR)
    dataset.save_to_disk(DATASET_DIR)
