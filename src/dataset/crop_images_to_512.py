#!/usr/bin/env python3
"""
データセット内の全ての_imag.jpgファイルから余計な白い領域を取り除き、
512x512の画像として保存し直すスクリプト
"""

import os
import glob
from PIL import Image
from tqdm import tqdm


def crop_image_to_512(image_path, backup=True):
    """
    1000x800の画像から512x512の領域を切り出す

    キャンバス座標系から画像座標系への変換:
    - キャンバス座標(244, 144)が画像座標(0, 0)に対応
    - 512x512の領域を切り出す

    Args:
        image_path: 画像ファイルのパス
        backup: 元画像をバックアップするかどうか
    """
    try:
        # 画像を読み込む
        img = Image.open(image_path)
        width, height = img.size

        # 既に512x512の場合はスキップ
        if width == 512 and height == 512:
            print(f"  Already 512x512, skipping: {os.path.basename(image_path)}")
            return False

        # 1000x800の画像から512x512を切り出す
        # キャンバス座標(244, 144)が画像座標(0, 0)になるように切り出し
        # つまり、左上座標(244, 144)から512x512の領域を切り出す
        left = 244
        top = 144
        right = left + 512
        bottom = top + 512

        # 切り出し
        cropped_img = img.crop((left, top, right, bottom))

        # バックアップを作成
        if backup:
            backup_path = image_path.replace('.jpg', '_original.jpg')
            if not os.path.exists(backup_path):
                img.save(backup_path)
                print(f"  Backup saved: {os.path.basename(backup_path)}")

        # 切り出した画像を保存（元のファイルを上書き）
        cropped_img.save(image_path, quality=95)

        print(f"  Cropped: {os.path.basename(image_path)} ({width}x{height} -> 512x512)")
        return True

    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return False


def main():
    import sys

    # コマンドライン引数からディレクトリを取得（引数がない場合はpixartをデフォルト）
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    else:
        dataset_dir = "/home/okumura/lab/grad_thesis_vp/vanishing_point/dataset_for_inference_pixart"

    # 全ての_imag.jpgファイルを取得
    image_files = glob.glob(os.path.join(dataset_dir, "*_imag.jpg"))

    if not image_files:
        print("No _imag.jpg files found in the dataset directory.")
        return

    print(f"Found {len(image_files)} _imag.jpg files to process.")
    print(f"Processing images in: {dataset_dir}")
    print("-" * 50)

    processed_count = 0
    skipped_count = 0
    error_count = 0

    # 各画像を処理
    for image_path in tqdm(image_files, desc="Cropping images"):
        result = crop_image_to_512(image_path, backup=True)
        if result:
            processed_count += 1
        elif result is False:
            skipped_count += 1
        else:
            error_count += 1

    # 結果を表示
    print("-" * 50)
    print("Processing complete!")
    print(f"  Processed: {processed_count} images")
    print(f"  Skipped (already 512x512): {skipped_count} images")
    if error_count > 0:
        print(f"  Errors: {error_count} images")

    # バックアップファイルの存在確認
    backup_files = glob.glob(os.path.join(dataset_dir, "*_original.jpg"))
    if backup_files:
        print(f"\nBackup files created: {len(backup_files)}")
        print("To restore original files, rename *_original.jpg back to *_imag.jpg")


if __name__ == "__main__":
    main()