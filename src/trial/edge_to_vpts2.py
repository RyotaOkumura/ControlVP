import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.additional_loss import AdditionalLossCalculator

FOCAL_LENGTH = 1
IMAGE_SIZE = 512


# 3D座標から2D座標への変換 (学習でつかっているものと同一)
def vpt3d_to_2d(w):
    x = w[0] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
    y = -w[1] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
    return x, y


# エッジマップから消失点に向かうエッジの強度を計算する (学習でつかっているものとほぼ同一だが、返り値の種類が多い)
def calc_edge_to_vanishing_point(edge_map, vanishing_points):
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
        edge_dx = edge_map_b[0]
        edge_dy = edge_map_b[1]
        edge_magnitude = torch.norm(edge_map_b, p=2, dim=0)
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

            # パラメトリック方程式のパラメータtを計算
            # (x, y) = (x0, y0) + t * (dy, -dx) ←(x0, y0)を通る方向ベクトル(dy, -dx)の直線
            # (x - vp_x)^2 + (y - vp_y)^2 = 10^2 ←消失点からの距離が10の円
            # 直線と円の交点が存在する条件は以下を満たすtが存在すること。
            # (x0 + t*(dy) - vp_x)^2 + (y0 + t*(-dx) - vp_y)^2 = 10^2

            a = edge_dx.pow(2) + edge_dy.pow(2)  # t^2の係数
            b = 2 * (
                (x_valid - vp_x) * edge_dy + (y_valid - vp_y) * (-edge_dx)
            )  # tの係数
            c = (
                (x_valid - vp_x).pow(2) + (y_valid - vp_y).pow(2) - 10**2
            )  # 定数項（半径10の二乗）
            # 判別式
            discriminant = b.pow(2) - 4 * a * c
            # print(f"discriminant: {discriminant}")

            # バイナリな判定の代わりにシグモイド関数で滑らかに遷移
            temperature = 10.0  # シグモイドの急峻さを調整
            valid_edges = torch.sigmoid(temperature * discriminant)
            # エッジの強度も考慮
            edge_weights = torch.mul(edge_magnitude, valid_edges)
            batch_total += torch.sum(edge_weights)
            # print(f"vp_x: {vp_x}, vp_y: {vp_y}, batch_total: {batch_total}")
        result[batch_idx] = (
            batch_total / valid_vp_count / H / W if valid_vp_count > 0 else 0
        )
    return result, edge_weights


def main(path_img, path_vpts):
    # メイン処理

    # 消失点データの読み込み
    vpts_data = np.load(path_vpts)
    vpts_3d = vpts_data["vpts"]
    # vpts_2d = torch.tensor([[vpt3d_to_2d(vpts_3d[0])]])
    vpts_2d = torch.tensor([[[300, 400]]])

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
    plt.savefig("edges_to_vpts.png")


if __name__ == "__main__":
    path_img = "/srv/datasets3/HoliCity/images/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_imag.jpg"
    path_vpts = "/srv/datasets3/HoliCity/vanishing_points/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_vpts.npz"
    main(path_img, path_vpts)
