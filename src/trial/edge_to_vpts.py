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


def visualize_edges_to_vp(edge_map, vp, angle_threshold=0.866):  # cos(30度) ≒ 0.866
    """
    特定の消失点に向かうエッジを抽出して可視化
    edge_map: [2, H, W] のエッジマップ
    vp: [x, y] の消失点座標
    angle_threshold: 消失点方向との内積の閾値（デフォルトは30度）
    """
    H, W = edge_map.shape[1:]
    device = edge_map.device

    # エッジの方向成分を個別に保存
    edge_y_component = edge_map[0].cpu().numpy()  # dy
    edge_x_component = -edge_map[1].cpu().numpy()  # -dx

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

    # エッジの方向ベクトル（正規化）
    # 勾配の法線方向が直線の方向
    edge_dx = edge_map[0][edge_mask]  # dy
    edge_dy = -edge_map[1][edge_mask]  # -dx
    edge_norms = torch.sqrt(edge_dx**2 + edge_dy**2) + 1e-6
    edge_dx = edge_dx / edge_norms
    edge_dy = edge_dy / edge_norms

    # 消失点への方向ベクトル（正規化）
    vp_dy = vp[1] - y_valid
    vp_dx = vp[0] - x_valid
    vp_norms = torch.sqrt(vp_dx**2 + vp_dy**2) + 1e-6
    vp_dx = vp_dx / vp_norms
    vp_dy = vp_dy / vp_norms

    # 方向ベクトル間の内積を計算（cosθ）
    cos_theta = torch.abs(vp_dx * edge_dx + vp_dy * edge_dy)

    # 内積が閾値以上のエッジを抽出
    valid_edges = cos_theta > angle_threshold

    # 結果の可視化用マスク作成
    result_mask = torch.zeros((H, W), device=device)
    result_mask[y_valid[valid_edges], x_valid[valid_edges]] = edge_magnitude[edge_mask][
        valid_edges
    ]

    return result_mask.cpu().numpy(), edge_x_component, edge_y_component


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
plt.subplot(151)
plt.imshow(image)
plt.title("Original Image")

# エッジ強度マップ
plt.subplot(152)
plt.imshow(edge_magnitude.cpu(), cmap="hot")
plt.title("Edge Magnitude Map")

# 1つ目の消失点に向かうエッジ
vp1_x, vp1_y = vpt3d_to_2d(vpts_3d[0])
edges_to_vp1, edge_x, edge_y = visualize_edges_to_vp(
    edges, torch.tensor([vp1_x, vp1_y])
)
plt.subplot(153)
plt.imshow(edge_x, cmap="bwr")  # bwrカラーマップを使用して正負を区別
plt.title("Edge X Component")

plt.subplot(154)
plt.imshow(edge_y, cmap="bwr")
plt.title("Edge Y Component")

# 消失点方向のエッジ
plt.subplot(155)
plt.imshow(edges_to_vp1, cmap="hot")
plt.title(f"Edges to VP1 ({vp1_x:.1f}, {vp1_y:.1f})")

plt.tight_layout()
plt.savefig("edges_to_vpts.png")
