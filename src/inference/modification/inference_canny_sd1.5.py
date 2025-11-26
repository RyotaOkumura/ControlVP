# !pip install opencv-python transformers accelerate
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from datasets import load_from_disk
import numpy as np
import torch
from PIL import Image
import cv2
from datetime import datetime
import os


def overlay_vanishing_point(image, vanishing_points):
    """
    消失点から放射状に直線を引いてimageに重ねる
    image: PIL Image
    vanishing_points: np.array shape=(N, 2) Nは消失点の数
    """
    # 画像をnumpy配列に変換
    img_array = np.array(image)

    # 画像サイズを取得
    h, w = img_array.shape[:2]

    # 角度のステップ（10度ごと）
    angle_step = 5
    angles = np.arange(0, 360, angle_step)

    # 各消失点について処理
    for vp in vanishing_points:
        # 各角度について直線を描画
        for angle in angles:
            theta = np.deg2rad(angle)
            direction = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)

            t = np.linspace(0, max(h, w) * 5, 100)
            points = vp.reshape(2, 1) + direction * t

            # 画像内の点のみをフィルタリング
            mask = (
                (points[0] >= 0) & (points[0] < w) & (points[1] >= 0) & (points[1] < h)
            )
            points = points[:, mask]

            # 直線を描画
            if points.shape[1] >= 2:
                for j in range(points.shape[1] - 1):
                    p1 = tuple(map(int, points[:, j]))
                    p2 = tuple(map(int, points[:, j + 1]))
                    cv2.line(img_array, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), 1)

    # numpy配列をPIL Imageに戻す
    image_with_lines = Image.fromarray(img_array)
    return image_with_lines


def main(
    target_image,
    condition_image,
    model_name,
    guidance_scale,
    controlnet_conditioning_scale,
    controlnet_guidance_scale,
    seed=None,
):
    # load control net and stable diffusion v1-5
    controlnet = ControlNetModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        # use_safetensors=True,
    )
    # controlnet = ControlNetModel.from_pretrained(
    #     "/home/okumura/lab/vanishing_point/src/model_out/checkpoint-480000/controlnet",
    #     torch_dtype=torch.float16,
    # )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        # "stabilityai/stable-diffusion-2-1",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    # speed up diffusion process with faster scheduler and memory optimization
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed
    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()

    # ControlNetを使用した画像生成
    # generator = torch.manual_seed(seed)
    controlnet_images = pipe(
        prompt="buildings with brown bricks on both sides of the road in London, high quality, photorealistic",
        # "painting of a girl",
        image=condition_image,
        # num_inference_steps=20,
        # generator=generator,
        num_images_per_prompt=5,
        guidance_scale=guidance_scale,
        controlnet_guidance_scale=controlnet_guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{os.path.dirname(__file__)}/output/raw_controlnet"
    os.makedirs(output_dir, exist_ok=True)

    # グリッド用の画像リストを作成
    num_images = len(controlnet_images)
    grid_images = []

    # 1行目: target_imageとcondition_image
    grid_images.extend([target_image, condition_image])

    # 2行目以降: controlnet_images
    for controlnet_img in controlnet_images:
        grid_images.append(controlnet_img)

    # グリッドの作成
    rows = 1 + (num_images + 1) // 2  # target/conditionの行 + 生成画像の行
    cols = 2
    cell_size = 512  # 画像サイズ

    # 空の画像を作成
    grid = Image.new("RGB", (cell_size * cols, cell_size * rows))

    # グリッドに画像を配置
    for idx, img in enumerate(grid_images):
        row = idx // cols
        col = idx % cols
        grid.paste(img, (col * cell_size, row * cell_size))

    # 個別画像を保存
    for i, img in enumerate(controlnet_images):
        img.save(
            f"{output_dir}/{timestamp}_canny_gs-{controlnet_guidance_scale}_image-{i}.png"
        )
        print(
            f"individual image {i} saved to {output_dir}/{timestamp}_canny_gs-{controlnet_guidance_scale}_image-{i}.png"
        )

    # グリッド画像を保存
    grid.save(
        f"{output_dir}/{timestamp}_canny_gs-{controlnet_guidance_scale}_comparison.png"
    )
    print(
        f"output image saved to {output_dir}/{timestamp}_canny_gs-{controlnet_guidance_scale}_comparison.png"
    )


if __name__ == "__main__":
    idx = 20
    # model_name = "/home/okumura/lab/grad_thesis_vp/vanishing_point/src/train/model_out_contour_vp_loss_w_tog_loss/checkpoint-25500/controlnet"
    # model_name = "/home/okumura/lab/grad_thesis_vp/vanishing_point/ckpt/contour/successful/model_out_contour_vp_loss_w-1000_v-pred/checkpoint-25500/controlnet"
    # model_name = "/home/okumura/lab/grad_thesis_vp/vanishing_point/src/train/model_out_contour_vp_loss_sd2-base/checkpoint-25000/controlnet"
    # model_name = "/home/okumura/lab/grad_thesis_vp/vanishing_point/src/train/model_out_contour_vp_loss_sd2-base_w-10/checkpoint-30000/controlnet"
    # model_name = "thibaud/controlnet-sd21-canny-diffusers"
    model_name = "lllyasviel/sd-controlnet-canny"

    dataset_path = "/srv/datasets3/HoliCity/dataset_w_vpts_edges"
    dataset = load_from_disk(dataset_path)
    target_image = dataset[idx]["image"]

    # # edgesから条件画像を作成
    # edges = dataset[idx]["edge"]
    # condition_image = np.zeros((512, 512, 3), dtype=np.uint8)
    # if edges:  # エッジが存在する場合
    #     for vp_edges in edges:
    #         for i in range(0, len(vp_edges), 4):
    #             x1, y1 = int(vp_edges[i]), int(vp_edges[i + 1])
    #             x2, y2 = int(vp_edges[i + 2]), int(vp_edges[i + 3])
    #             cv2.line(
    #                 condition_image, (x1, y1), (x2, y2), (255, 255, 255), 1
    #             )  # 白線で描画
    # condition_image = Image.fromarray(condition_image)

    # target_image = load_image(
    #     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/non-enhanced-prompt.png"
    # )
    low_threshold = 100
    high_threshold = 200
    condition_image = cv2.Canny(np.array(target_image), low_threshold, high_threshold)
    condition_image = condition_image[:, :, None]
    condition_image = np.concatenate(
        [condition_image, condition_image, condition_image], axis=2
    )
    # uint8型に変換（0-255の整数範囲）
    condition_image = condition_image.astype(np.uint8)

    condition_image = Image.fromarray(condition_image)

    guidance_scale = 7.5
    controlnet_conditioning_scale = 1.0
    controlnet_guidance_scale = 3.0
    # seed = 4
    main(
        target_image,
        condition_image,
        model_name,
        guidance_scale,
        controlnet_conditioning_scale,
        controlnet_guidance_scale,
        # seed,
    )
