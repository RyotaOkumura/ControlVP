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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from src.pipelines import StableDiffusionControlNetInpaintCFGPipeline


def create_result_grid(image_rows, base_size=None):
    """Create result image in grid format

    Args:
        image_rows (list[list[PIL.Image]]): 2D array of images. Each row can contain different number of images
        base_size (tuple[int, int], optional): Size of blank image. If None, use the size of the first image

    Returns:
        PIL.Image: Result image in grid format
    """
    if not image_rows or not image_rows[0]:
        raise ValueError("Image array is empty")

    # Determine the base size
    if base_size is None:
        # Use the size of the first image
        base_size = image_rows[0][0].size

    # Create blank image
    blank_image = Image.new("RGB", base_size, (255, 255, 255))

    # Calculate the maximum number of columns
    max_cols = max(len(row) for row in image_rows)

    # Fill each row with blank image to match the maximum number of columns
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
    # Read edges two by two
    condition_image = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(0, len(edges), 4):
        x1, y1 = int(edges[i]), int(edges[i + 1])
        x2, y2 = int(edges[i + 2]), int(edges[i + 3])
        cv2.line(
            condition_image, (x1, y1), (x2, y2), (255, 255, 255), line_width
        ) 
    return condition_image


def main(init_image_path, condition_npz_path, mask_image_path=None):
    """
    Args:
        init_image_path (str): Path to initial image
        condition_npz_path (str): Path to condition image npz file
        mask_image_path (str, optional): Path to mask image. If None, generate automatically
    """
    # load models
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_PATH,
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

    # Prepare mask image
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
        control_image=control_image,
        strength=STRENGTH,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        controlnet_guidance_scale=CONTROLNET_GUIDANCE_SCALE,
        # num_inference_steps=20,
        # guidance_scale=7.5,
        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
    ).images
    condition_for_overlay = make_condition_image(condition_npz_path, line_width=2)
    print(pipeline.unet.config.in_channels)

    # Set output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{os.path.dirname(__file__)}/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = ""
    if mask_image_path is None:
        output_path = f"{output_dir}/{timestamp}_str-{STRENGTH}_blur-{BLUR_FACTOR}_kernel-{KERNEL_SIZE}.jpg"
    else:
        output_path = f"{output_dir}/{timestamp}_w-mask_str-{STRENGTH}_blur-{BLUR_FACTOR}_kernel-{KERNEL_SIZE}.jpg"

    # Save grid image
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
            ]
        )

        grid_image.save(output_path)
        print(f"output image saved to {output_path}")

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
    mask_image = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_image[condition_image == 255] = 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask_image, kernel, iterations=1)

    return dilated_mask


if __name__ == "__main__":
    SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "sample_data")
    IMAGE_PATHS = [
        {
            "init": f"{SAMPLE_DATA_DIR}/data_20250328_192600_imag.jpg",
            "condition": f"{SAMPLE_DATA_DIR}/data_20250328_192600_edge.npz",
            "mask": f"{SAMPLE_DATA_DIR}/data_20250328_192600_mask.jpg",
        },
        {
            "init": f"{SAMPLE_DATA_DIR}/data_20250324_054933_imag.jpg",
            "condition": f"{SAMPLE_DATA_DIR}/data_20250324_054933_edge.npz",
            "mask": f"{SAMPLE_DATA_DIR}/data_20250324_054933_mask.jpg",
        },
        {
            "init": f"{SAMPLE_DATA_DIR}/data_20250324_061130_imag.jpg",
            "condition": f"{SAMPLE_DATA_DIR}/data_20250324_061130_edge.npz",
            "mask": f"{SAMPLE_DATA_DIR}/data_20250324_061130_mask.jpg",
        },
        {
            "init": f"{SAMPLE_DATA_DIR}/data_20250521_043108_imag.jpg",
            "condition": f"{SAMPLE_DATA_DIR}/data_20250521_043108_edge.npz",
            "mask": f"{SAMPLE_DATA_DIR}/data_20250521_043108_mask.jpg",
        },
        {
            "init": f"{SAMPLE_DATA_DIR}/data_20250503_162346_256_232_imag.jpg",
            "condition": f"{SAMPLE_DATA_DIR}/data_20250503_162346_256_232_edge.npz",
            "mask": f"{SAMPLE_DATA_DIR}/data_20250503_162346_256_232_mask.jpg",
        },
        {
            "init": f"{SAMPLE_DATA_DIR}/data_20250504_134743_242_390_imag.jpg",
            "condition": f"{SAMPLE_DATA_DIR}/data_20250504_134743_242_390_edge.npz",
            "mask": f"{SAMPLE_DATA_DIR}/data_20250504_134743_242_390_mask.jpg",
        },
    ]

    CONTROLNET_MODEL_PATH = os.path.join(PROJECT_ROOT, "ckpts/controlvp_controlnet")
    MODEL_NAME = "stabilityai/stable-diffusion-2-inpainting"

    # Other parameters
    KERNEL_SIZE = 30
    BLUR_FACTOR = 20
    STRENGTH = 1.0
    NUM_IMAGES_PER_PROMPT = 10
    PROMPT = "building, high quality, photorealistic"
    SAVE_EACH_IMAGE = True
    SAVE_GRID_IMAGE = True
    CONTROLNET_CONDITIONING_SCALE = 1.0
    CONTROLNET_GUIDANCE_SCALE = 3.0

    # Execute with index 0 of image set
    image_set_indexes = [0]
    paths = [IMAGE_PATHS[image_set_index] for image_set_index in image_set_indexes]

    for i in range(len(image_set_indexes)):
        for _ in range(1):
            main(paths[i]["init"], paths[i]["condition"], paths[i]["mask"])
