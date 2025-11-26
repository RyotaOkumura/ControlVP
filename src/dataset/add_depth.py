from transformers import pipeline
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# load pipe
pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
)

DATASET_DIR = "./dataset_for_inference_pixart"

for f in os.listdir(DATASET_DIR):
    if f.endswith("_imag.jpg"):
        # load image
        image_path = os.path.join(DATASET_DIR, f)
        image = Image.open(image_path)
        file_name = f.replace("_imag.jpg", "_depth.npy")

        # inference
        depth = pipe(image)["depth"]

        # save depth map as numpy array
        np.save(os.path.join(DATASET_DIR, file_name), depth)
