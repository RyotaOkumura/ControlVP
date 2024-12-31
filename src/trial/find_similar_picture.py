from datasets import load_from_disk
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os


def load_target_image(image_path: str) -> np.ndarray:
    target_img = Image.open(image_path)
    return np.array(target_img)


def find_most_similar_image(dataset_path: str, target_image_path: str):
    dataset = load_from_disk(dataset_path)
    target_img = load_target_image(target_image_path)

    best_similarity = -float("inf")
    best_idx = -1

    # データセット内の各画像と比較
    for i, data in enumerate(dataset):
        current_img = np.array(data["image"])

        # 画像のサイズを合わせる
        if current_img.shape != target_img.shape:
            current_img = np.array(
                Image.fromarray(current_img).resize(
                    (target_img.shape[1], target_img.shape[0])
                )
            )

        # SSIMで類似度を計算
        similarity = ssim(target_img, current_img, channel_axis=2)

        if similarity > best_similarity:
            best_similarity = similarity
            best_idx = i

        if i % 100 == 0:  # 進捗表示
            print(
                f"Processed {i} images. Current best similarity: {best_similarity}, best_idx: {best_idx}"
            )

    # 最も類似度の高い画像を保存
    best_match = dataset[best_idx]["image"]
    best_match.save("most_similar_image.png")
    print(f"Best match found at index {best_idx} with similarity {best_similarity}")
    return best_idx, best_similarity


if __name__ == "__main__":
    target_path = "/home/okumura/lab/vanishing_point/src/generated_images/step_5000_20241231_012116_batch0.png"
    dataset_path = "/srv/datasets3/HoliCity/dataset_w_vpts"
    find_most_similar_image(dataset_path, target_path)
