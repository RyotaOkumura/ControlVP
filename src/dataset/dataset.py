from torch.utils.data import Dataset
from PIL import Image
import os
import json
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config


def create_hf_dataset_structure(image_dir, conditioning_dir, output_dir):
    """
    既存のデータセット構造をHugging Face形式に変換します。

    Args:
        image_dir (str): 元の画像ディレクトリのパス
        conditioning_dir (str): 元の条件画像ディレクトリのパス
        output_dir (str): 出力先ディレクトリのパス
    """
    # 出力ディレクトリ構造の作成
    train_dir = os.path.join(output_dir, "data", "train")
    image_output_dir = os.path.join(train_dir, "image")
    conditioning_output_dir = os.path.join(train_dir, "conditioning_image")

    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(conditioning_output_dir, exist_ok=True)

    # 画像ファイルのリストを取得
    image_files = sorted(os.listdir(image_dir))

    # metadata.jsonlの作成
    metadata_entries = []

    for img_file in image_files:
        # 対応する条件画像が存在することを確認
        if not os.path.exists(os.path.join(conditioning_dir, img_file)):
            print(f"Warning: No matching conditioning image for {img_file}")
            continue

        # 画像をコピー
        shutil.copy2(
            os.path.join(image_dir, img_file), os.path.join(image_output_dir, img_file)
        )
        shutil.copy2(
            os.path.join(conditioning_dir, img_file),
            os.path.join(conditioning_output_dir, img_file),
        )

        # メタデータエントリの作成
        entry = {
            "image": f"image/{img_file}",
            "conditioning_image": f"conditioning_image/{img_file}",
            "text": "",  # 空のテキストプロンプト
        }
        metadata_entries.append(entry)

    # metadata.jsonlの書き出し
    metadata_path = os.path.join(train_dir, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # dataset_dict.jsonの作成
    dataset_dict = {"train": {"filename": "data/train/metadata.jsonl"}}

    with open(
        os.path.join(output_dir, "dataset_dict.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(dataset_dict, f, indent=2, ensure_ascii=False)

    print(f"Dataset created at: {output_dir}")
    print(f"Total images processed: {len(metadata_entries)}")


if __name__ == "__main__":
    # 使用例
    create_hf_dataset_structure(
        image_dir="path/to/your/image_dir",
        conditioning_dir="path/to/your/conditioning_dir",
        output_dir="path/to/output/dataset",
    )
