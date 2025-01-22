import torch
import numpy as np
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers.utils import make_image_grid
import PIL
import cv2
import os
from datetime import datetime


# load ControlNet
def main(
    controlnet_model_path,
    model_name,
    init_image,
    condition_image,
    mask_image,
    prompt,
    strength=1.0,
    blur_factor=0,
):
    """
    Args:
        controlnet_model_path (str): ControlNetのパス
        model_name (str): StableDiffusionのパス
        init_image (PIL.Image): 初期画像
        condition_image (numpy.ndarray): 条件画像、shape=(512, 512, 3)、dtype=uint8
        mask_image (numpy.ndarray): マスク画像、shape=(512, 512, 3)、dtype=uint8
        blur_factor (int, optional): マスクのぼかし係数。デフォルト値は0
    """
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_path,
        torch_dtype=torch.float16,
    )

    # pass ControlNet to the pipeline
    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        model_name,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipeline.enable_model_cpu_offload()
    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    pipeline.enable_xformers_memory_efficient_attention()

    blurred_mask = pipeline.mask_processor.blur(
        PIL.Image.fromarray(mask_image), blur_factor=blur_factor
    )
    # control_imageを正しい形状に変換（HWC -> CHW）
    control_image = condition_image.copy()  # numpy.ndarray (H, W, C)
    control_image = PIL.Image.fromarray(control_image)  # PIL.Image に変換

    image = pipeline(
        prompt=prompt,
        image=init_image,
        mask_image=blurred_mask,
        control_image=control_image,  # PIL.Image として渡す
        strength=strength,
        # num_inference_steps=20,
        # guidance_scale=7.5,
        # controlnet_conditioning_scale=1.0,
    ).images[0]

    grid_image = make_image_grid(
        [
            init_image,
            blurred_mask,
            control_image,  # すでにPIL.Image
            image,
        ],
        rows=2,
        cols=2,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        f"{os.path.dirname(__file__)}/output/inpainting/{timestamp}_image_grid.png"
    )
    grid_image.save(output_path)
    print(f"output image saved to {output_path}")


def make_mask(condition_image, kernel_size=100):
    # condition_imageの白い線の周りだけマスクする
    mask_image = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_image[condition_image == 255] = 255

    # マスクを膨張させて線の周りをマスクする
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask_image, kernel, iterations=1)

    return dilated_mask


def make_condition_image(condition_npz_path):
    edges = np.load(condition_npz_path)["edges"]
    # edgesを二要素ずつ読み込む
    condition_image = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(0, len(edges), 4):
        x1, y1 = int(edges[i]), int(edges[i + 1])
        x2, y2 = int(edges[i + 2]), int(edges[i + 3])
        cv2.line(condition_image, (x1, y1), (x2, y2), (255, 255, 255), 1)  # 白線で描画
    return condition_image


if __name__ == "__main__":
    controlnet_model_name = "/home/okumura/lab/vanishing_point/ckpt/model_out_w_vpts_edges_black-bg2/checkpoint-3500/controlnet"
    model_name = "stabilityai/stable-diffusion-2-1"
    init_image_path = (
        "/home/okumura/lab/vanishing_point/data/data_20250121_184745_imag.jpg"
    )
    init_image = PIL.Image.open(init_image_path)
    condition_npz_path = (
        "/home/okumura/lab/vanishing_point/data/data_20250121_184745_edge.npz"
    )
    condition_image = make_condition_image(condition_npz_path)
    mask_image = make_mask(condition_image, kernel_size=80)
    prompt = "modern buildings, high quality, photorealistic"
    main(
        controlnet_model_name,
        model_name,
        init_image,
        condition_image,
        mask_image,
        prompt,
        strength=0.7,
        blur_factor=50,
    )
