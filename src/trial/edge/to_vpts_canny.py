import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import cv2

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from src.additional_loss_kornia import AdditionalLossCalculatorKornia

FOCAL_LENGTH = 1
IMAGE_SIZE = 512

# cosθの値が閾値以上のエッジを抽出する


# 3D座標から2D座標への変換 (学習でつかっているものと同一)
def vpt3d_to_2d(w):
    x = w[0] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
    y = -w[1] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
    return x, y


# AdditionalLossCalculatorKorniaのcalc_scores関数を関数に置き換える
def calc_scores(
    self, edges, magnitudes, vanishing_points, angle_threshold=0.0, eps=1e-8
):
    """
    エッジマップのうち消失点に向かうものを計算する
    edges: [B, C, 2, H, W]
    magnitudes: [B, C, H, W]
    vanishing_points: [B, 3, 2]
    returns: [B]
    """
    B, C, _, H, W = edges.shape
    device = edges.device
    result = torch.zeros(B, device=device)
    gx = edges[:, :, 0]
    gy = edges[:, :, 1]
    magnitude = magnitudes
    dir_xs = -gy / magnitude  # [B, C, H, W]
    dir_ys = gx / magnitude  # [B, C, H, W]]
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    for batch in range(B):
        batch_total = 0
        dir_x = dir_xs[batch]
        dir_y = dir_ys[batch]
        vps = vanishing_points[batch]
        valid_vp_count = 0
        for vp in vps:
            vp_x, vp_y = vp[0].item(), vp[1].item()
            is_valid_vp = ~((vp[0] == -1) & (vp[1] == -1))
            valid_vp_count += is_valid_vp.float()
            vp_dx = vp_x - x_coords
            vp_dy = vp_y - y_coords
            vp_norms = torch.sqrt(vp_dx * vp_dx + vp_dy * vp_dy + eps)
            vp_dx = vp_dx / vp_norms
            vp_dy = vp_dy / vp_norms
            cos_theta = torch.abs(dir_x * vp_dx + dir_y * vp_dy)
            theta = torch.acos(torch.clamp(cos_theta, -0.999999, 0.999999))
            angle_threshold = torch.tensor(
                angle_threshold * np.pi / 180.0, device=device
            )
            temperature = 5.0
            valid_edges = 2 * torch.sigmoid(temperature * (angle_threshold - theta))
            edge_weights = magnitude * valid_edges
            batch_total += torch.sum(edge_weights) * is_valid_vp.float()
        result[batch] = batch_total / (valid_vp_count + eps) / H / W
    return result, edge_weights


AdditionalLossCalculatorKornia.calc_scores = calc_scores


def main(path_img, path_vpts=None, vpts_2d=None):
    # メイン処理
    if vpts_2d is None:
        # 消失点データの読み込み
        vpts_data = np.load(path_vpts)
        vpts_3d = vpts_data["vpts"]
        vpts_2d = torch.tensor([[vpt3d_to_2d(vpts_3d[0])]])
    else:
        vpts_2d = vpts_2d

    # 画像とエッジの読み込み
    image = Image.open(path_img)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image_tensor = transform(image).unsqueeze(0)  # [1, C, H, W]

    # エッジ検出
    additional_loss = AdditionalLossCalculatorKornia()
    cannys = additional_loss.detect_canny_edges(image_tensor)
    edges, magnitudes = additional_loss.detect_sobel_edges(cannys, eps=1e-8)
    gx = edges[:, :, 0]
    gy = edges[:, :, 1]
    result, edge_weights = additional_loss.calc_scores(
        edges, magnitudes, vpts_2d
    )  # [1, 2, H, W]
    print(result[0])
    # 可視化
    plt.figure(figsize=(20, 5))

    # 元画像
    plt.subplot(141)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(142)
    plt.imshow(gx[0][0].cpu(), cmap="bwr")
    plt.title("Edge X Component")

    plt.subplot(143)
    plt.imshow(gy[0][0].cpu(), cmap="bwr")
    plt.title("Edge Y Component")

    # エッジ強度マップ
    plt.subplot(144)
    plt.imshow(edge_weights[0][0].cpu(), cmap="hot")
    plt.title("Edge Map in Direction to VP")

    plt.tight_layout()
    plt.savefig("src/trial/images/to_vpts3.png")
    return result[0]


if __name__ == "__main__":
    path_img = "/srv/datasets3/HoliCity/images/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_imag.jpg"
    path_vpts = "/srv/datasets3/HoliCity/vanishing_points/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_vpts.npz"
    vpts_2d = torch.tensor([[[400.0, 300.0]]])
    gt = main(path_img, path_vpts)

    # # 他の座標でグリッドサーチし、gtを超えるものを抽出する
    # for x in range(-512, 1024, 128):
    #     for y in range(-512, 1024, 128):
    #         vpts_2d = torch.tensor([[[float(x), float(y)]]])
    #         score = main(path_img, path_vpts, vpts_2d)
    #         if score > gt:
    #             print(f"x: {x}, y: {y}, score: {score}")
