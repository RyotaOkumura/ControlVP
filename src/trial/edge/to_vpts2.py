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
from src.additional_loss import AdditionalLossCalculator

FOCAL_LENGTH = 1
IMAGE_SIZE = 512

# cosθの値が閾値以上のエッジを抽出する


# 3D座標から2D座標への変換 (学習でつかっているものと同一)
def vpt3d_to_2d(w):
    x = w[0] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
    y = -w[1] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
    return x, y


def calc_edge_to_vanishing_point(edge_map, vanishing_points, angle_threshold=0.0):
    """
    エッジマップのうち消失点に向かうものを計算する
    edge_map: [B, 2, H, W]
    vanishing_points: [B, 3, 2]
    returns: [B]
    """
    B, _, H, W = edge_map.shape
    device = edge_map.device
    result = torch.zeros(B, device=device)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )

    # バッチ内の各サンプルに対して処理
    for batch_idx in range(B):
        batch_total = 0
        y_valid = y_coords
        x_valid = x_coords
        edge_map_b = edge_map[batch_idx]
        edge_magnitude = torch.norm(edge_map_b, p=2, dim=0).clamp(min=1e-6)
        edge_dx = edge_map_b[0] / edge_magnitude
        edge_dy = edge_map_b[1] / edge_magnitude
        dir_x = edge_dy
        dir_y = edge_dx
        vps = vanishing_points[batch_idx].to(device)  # 現在のデバイスに移動

        # print(f"vps: {vps.shape}")
        valid_vp_count = 0
        for n in range(vps.shape[0]):  # 消失点の数でループ
            vp = vps[n]
            vp_x, vp_y = vp[0].item(), vp[1].item()  # テンソルから数値に変換
            # print(f"vp: {vp_x}, {vp_y}")
            if vp_x == -1 and vp_y == -1:
                continue
            valid_vp_count += 1

            # 消失点への方向ベクトル（正規化）
            vp_dy = vp[1] - y_valid
            vp_dx = vp[0] - x_valid
            vp_norms = torch.norm(torch.stack([vp_dx, vp_dy]), p=2, dim=0).clamp(
                min=1e-6
            )
            vp_dx = vp_dx / vp_norms
            vp_dy = vp_dy / vp_norms

            # 方向ベクトル間の内積を計算（cosθ）
            cos_theta = torch.abs(dir_x * vp_dx + dir_y * vp_dy)
            # θを計算（ラジアン） 0-π
            theta = torch.acos(
                torch.clamp(cos_theta, -1.0, 1.0)
            )  # numerical stabilityのためclamp
            # nanが含まれるかチェック
            if torch.isnan(theta).any():
                print("thetaにnanが含まれています")
                continue
            # 角度の閾値
            angle_threshold = torch.tensor(
                angle_threshold * np.pi / 180.0, device=device
            )

            # temperatureを使用してシグモイド関数を適用
            temperature = 5.0
            valid_edges = 2 * torch.sigmoid(temperature * (angle_threshold - theta))
            # nanが含まれるかチェック
            if torch.isnan(valid_edges).any():
                print("valid_edgesにnanが含まれています")
                continue
            edge_weights = edge_magnitude * valid_edges
            batch_total += torch.sum(edge_weights)
        result[batch_idx] = (
            batch_total / valid_vp_count / H / W if valid_vp_count > 0 else 0
        )
    return result, edge_weights


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
    image = Image.open(path_img).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image_tensor = transform(image).unsqueeze(0)  # [1, C, H, W]

    # エッジ検出
    additional_loss = AdditionalLossCalculator()
    edges = additional_loss.detect_edges(image_tensor)  # [1, 2, H, W]
    edge_magnitude = torch.norm(edges[0], p=2, dim=0)
    edge_x = edges[0][0]
    edge_y = edges[0][1]

    # 消失点に向かうエッジの強度を計算
    result, edge_weights = calc_edge_to_vanishing_point(edges, vpts_2d)
    print(result[0])
    # 可視化
    plt.figure(figsize=(20, 5))

    # 元画像
    plt.subplot(151)
    plt.imshow(image)
    plt.title("Original Image")

    # エッジ強度マップ
    plt.subplot(152)
    plt.imshow(edge_magnitude.cpu(), cmap="hot")
    plt.title("Edge Magnitude Map")

    plt.subplot(153)
    plt.imshow(edge_x, cmap="bwr")  # bwrカラーマップを使用して正負を区別
    plt.title("Edge X Component")

    plt.subplot(154)
    plt.imshow(edge_y, cmap="bwr")
    plt.title("Edge Y Component")

    # 消失点方向のエッジ
    plt.subplot(155)
    plt.imshow(edge_weights * 10, cmap="hot")
    plt.title(
        f"Edges to VP1 ({float(vpts_2d[0][0][0]):.1f}, {float(vpts_2d[0][0][1]):.1f})"
    )

    plt.tight_layout()
    plt.savefig("to_vpts2.png")
    return result[0]


if __name__ == "__main__":
    path_img = "/srv/datasets3/HoliCity/images/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_imag.jpg"
    path_img = "/home/okumura/lab/vanishing_point/for_edge.png"
    path_vpts = "/srv/datasets3/HoliCity/vanishing_points/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_vpts.npz"
    vpts_2d = torch.tensor([[[180.0, 160.0]]])
    gt = main(path_img, path_vpts, vpts_2d)

    # # 他の座標でグリッドサーチし、gtを超えるものを抽出する
    # for x in range(-512, 1024, 128):
    #     for y in range(-512, 1024, 128):
    #         vpts_2d = torch.tensor([[[float(x), float(y)]]])
    #         score = main(path_img, path_vpts, vpts_2d)
    #         if score > gt:
    #             print(f"x: {x}, y: {y}, score: {score}")
