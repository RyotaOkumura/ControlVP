from datasets import Dataset
import os
from PIL import Image
import numpy as np
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
MIN_SAMPLES = 125


def create_dataset_from_images(image_base_dir, vpts_base_dir):
    """
    Create a dataset from image folder and vanishing points folder

    Args:
        image_base_dir (str): Base directory containing images
        vpts_base_dir (str): Base directory containing vanishing point data
    """
    image_paths = []
    vpts_paths = []
    # Recursively search subdirectories
    for root, _, files in os.walk(image_base_dir):
        for file in files:
            if file.endswith(("jpg", "jpeg", "png")):
                # Get image path
                img_path = os.path.join(root, file)

                # Build path for corresponding vanishing point file
                relative_path = os.path.relpath(root, image_base_dir)
                vpts_file = file.replace(
                    "_imag.jpg", "_vpts-w-edges.npz"
                )
                vpts_path = os.path.join(vpts_base_dir, relative_path, vpts_file)

                # Add to dataset only if vanishing point file exists
                if os.path.exists(vpts_path):
                    image_paths.append(img_path)
                    vpts_paths.append(vpts_path)
                    print(img_path)

    # Load images and conditioning
    images = [Image.open(path) for path in image_paths]
    captions = [""] * len(images)
    vanishing_points = [return_valid_vpts(path) for path in vpts_paths]

    # Load edge data
    edges_data = []
    for vpts_path in vpts_paths:
        vpts_data = np.load(vpts_path, allow_pickle=True)
        edges_array = vpts_data["edges"]
        vpts_3d = vpts_data["vpts"]
        vpts_confidences = vpts_data["confidence"]
        valid_indices = [
            i for i in range(min(3, len(vpts_3d))) if vpts_confidences[i] > MIN_SAMPLES
        ]
        # Convert edge data to a simpler format
        valid_edges = []
        for idx in valid_indices:
            edge_group = edges_array[idx]
            edge_coords = []
            for edge in edge_group:
                edge_coords.extend(
                    [
                        float(edge[0][0]),
                        float(edge[0][1]),  # start x, y
                        float(edge[1][0]),
                        float(edge[1][1]),  # end x, y
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
    Create conditioning image from vanishing point data
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


if __name__ == "__main__":
    from src.config import config

    IMAGE_BASE_DIR = config["dataset"]["image_base_dir"]
    VPTS_BASE_DIR = config["dataset"]["vpts_base_dir"]
    DATASET_DIR = config["dataset"]["dataset_dir"]
    os.makedirs(DATASET_DIR, exist_ok=True)
    dataset = create_dataset_from_images(IMAGE_BASE_DIR, VPTS_BASE_DIR)
    dataset.save_to_disk(DATASET_DIR)
