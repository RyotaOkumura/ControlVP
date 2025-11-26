from datasets import Dataset
import os
from PIL import Image
import numpy as np
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# import cv2

MIN_SAMPLES = 125


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
                    "_imag.jpg", "_vpts-w-edges.npz"
                )  # ファイル名のパターンを変換
                vpts_path = os.path.join(vpts_base_dir, relative_path, vpts_file)

                # 消失点ファイルが存在する場合のみデータセットに追加
                if os.path.exists(vpts_path):
                    image_paths.append(img_path)
                    vpts_paths.append(vpts_path)
                    print(img_path)

    # 画像とconditioningを読み込む
    images = [Image.open(path) for path in image_paths]
    captions = [""] * len(images)
    vanishing_points = [return_valid_vpts(path) for path in vpts_paths]

    # エッジデータを読み込む
    edges_data = []
    for vpts_path in vpts_paths:
        vpts_data = np.load(vpts_path, allow_pickle=True)
        edges_array = vpts_data["edges"]
        vpts_3d = vpts_data["vpts"]
        vpts_confidences = vpts_data["confidence"]
        valid_indices = [
            i for i in range(min(3, len(vpts_3d))) if vpts_confidences[i] > MIN_SAMPLES
        ]
        # エッジデータをより単純な形式に変換
        valid_edges = []
        for idx in valid_indices:
            # 各エッジグループを単純な座標のリストに変換
            edge_group = edges_array[idx]
            edge_coords = []
            for edge in edge_group:
                # 各エッジの始点と終点の座標をフラットなリストとして保存
                edge_coords.extend(
                    [
                        float(edge[0][0]),
                        float(edge[0][1]),  # 始点のx, y
                        float(edge[1][0]),
                        float(edge[1][1]),  # 終点のx, y
                    ]
                )
            valid_edges.append(edge_coords)
        edges_data.append(valid_edges)

    dataset = Dataset.from_dict(
        {
            "image": images,
            "caption": captions,
            "vanishing_point": vanishing_points,
            "edge": edges_data,
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
        if w[2] == 0:
            return float("inf"), float("inf")
        x = w[0] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
        y = -w[1] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
        return x, y

    valid_vpts = [
        vpt3d_to_2d(vpts_3d[i])
        for i in range(min(3, len(vpts_3d)))
        if vpts_confidences[i] > MIN_SAMPLES
    ]
    valid_vpts = valid_vpts[:3]

    return valid_vpts


# def make_conditioning_image(vpts_path):
#     """
#     消失点データから条件画像を作成する。各消失点に対して別々の条件画像を返す。
#     """
#     valid_vpts = return_valid_vpts(vpts_path)
#     # 消失点がない場合は白画像を返す
#     if not valid_vpts:
#         white_image = Image.new("RGB", (512, 512), "white")
#         return [white_image]  # リストとして返す

#     # npzファイルからエッジデータを読み込む
#     vpts_data = np.load(vpts_path)
#     edges_array = vpts_data["edges"]
#     vpts_3d = vpts_data["vpts"]
#     vpts_confidences = vpts_data["confidence"]

#     # 有効な消失点のインデックスを取得
#     valid_indices = [
#         i for i in range(min(3, len(vpts_3d))) if vpts_confidences[i] > MIN_SAMPLES
#     ]

#     # 各消失点に対して別々の条件画像を作成
#     condition_images = []
#     for vpt_idx in valid_indices:
#         condition_image = np.zeros((512, 512, 3), dtype=np.uint8)
#         condition_image.fill(255)  # 白背景
#         edges = edges_array[vpt_idx]
#         for edge in edges:
#             pt1 = tuple(map(int, edge[0]))
#             pt2 = tuple(map(int, edge[1]))
#             cv2.line(condition_image, pt1, pt2, (0, 0, 0), 1)  # 黒線で描画
#         condition_images.append(Image.fromarray(condition_image))

#     return condition_images


if __name__ == "__main__":
    from src.config import config

    IMAGE_BASE_DIR = config["dataset"]["image_base_dir"]
    VPTS_BASE_DIR = config["dataset"]["vpts_base_dir"]
    DATASET_DIR = config["dataset"]["dataset_dir"]
    os.makedirs(DATASET_DIR, exist_ok=True)
    dataset = create_dataset_from_images(IMAGE_BASE_DIR, VPTS_BASE_DIR)
    dataset.save_to_disk(DATASET_DIR)
