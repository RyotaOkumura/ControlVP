import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import cv2

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.additional_loss import AdditionalLossCalculator


def visualize_edges_to_vp(edge_map, vp, distance_threshold=100):
    """
    特定の消失点に向かうエッジを抽出して可視化
    edge_map: [2, H, W] のエッジマップ
    vp: [x, y] の消失点座標
    """
    H, W = edge_map.shape[1:]
    device = edge_map.device

    # 座標グリッドの作成
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )

    # エッジの強度とマスクの計算
    edge_magnitude = torch.sqrt(torch.sum(edge_map**2, dim=0))
    edge_mask = edge_magnitude > 1  # エッジの閾値

    # 有効なエッジの抽出
    y_valid = y_coords[edge_mask]
    x_valid = x_coords[edge_mask]
    edge_dx = edge_map[1][edge_mask]
    edge_dy = edge_map[0][edge_mask]

    # エッジの正規化
    edge_norms = torch.sqrt(edge_dx**2 + edge_dy**2)
    edge_dx = edge_dx / (edge_norms + 1e-6)
    edge_dy = edge_dy / (edge_norms + 1e-6)
    dir_x = edge_dy
    dir_y = -edge_dx
    # 消失点までの距離計算
    a = dir_x**2 + dir_y**2
    b = 2 * ((x_valid - vp[0]) * dir_x + (y_valid - vp[1]) * dir_y)
    c = (x_valid - vp[0]) ** 2 + (y_valid - vp[1]) ** 2 - distance_threshold**2

    # 判別式による交点の存在確認
    discriminant = b**2 - 4 * a * c
    valid_edges = discriminant > 0

    # 結果の可視化用マスク作成
    result_mask = torch.zeros((H, W), device=device)
    result_mask[y_valid[valid_edges], x_valid[valid_edges]] = edge_magnitude[edge_mask][
        valid_edges
    ]

    return result_mask.cpu().numpy()


# メイン処理
path_img = (
    "/srv/datasets3/HoliCity/images/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_imag.jpg"
)
path_vpts = "/srv/datasets3/HoliCity/vanishing_points/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_vpts.npz"

# 画像とエッジの読み込み
image = Image.open(path_img)
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0)

# エッジ検出
additional_loss = AdditionalLossCalculator()
edges = additional_loss.detect_edges(image_tensor)[0]  # [2, H, W]
edge_magnitude = torch.sqrt(torch.sum(edges**2, dim=0))
edge_magnitude = edge_magnitude * (edge_magnitude > 1)
# 消失点データの読み込みと2D座標への変換
vpts_data = np.load(path_vpts)
vpts_3d = vpts_data["vpts"]
FOCAL_LENGTH = 1
IMAGE_SIZE = 512


def vpt3d_to_2d(w):
    x = w[0] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
    y = -w[1] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
    return x, y


# 可視化
plt.figure(figsize=(20, 5))

# 元画像
plt.subplot(141)
plt.imshow(image)
plt.title("Original Image")

# エッジ強度マップ
plt.subplot(142)
plt.imshow(edge_magnitude.cpu(), cmap="hot")
plt.title("Edge Magnitude Map")

# 1つ目の消失点に向かうエッジ
vp1_x, vp1_y = vpt3d_to_2d(vpts_3d[0])
edges_to_vp1 = visualize_edges_to_vp(edges, torch.tensor([vp1_x, vp1_y]))
plt.subplot(143)
plt.imshow(edges_to_vp1, cmap="hot")
plt.title(f"Edges to VP1 ({vp1_x}, {vp1_y})")

# 2つ目の消失点に向かうエッジ
vp2_x, vp2_y = vpt3d_to_2d(vpts_3d[1])
edges_to_vp2 = visualize_edges_to_vp(edges, torch.tensor([vp2_x, vp2_y]))
plt.subplot(144)
plt.imshow(edges_to_vp2, cmap="hot")
plt.title(f"Edges to VP2 ({vp2_x}, {vp2_y})")

plt.tight_layout()
plt.savefig("edges_to_vpts.png")
