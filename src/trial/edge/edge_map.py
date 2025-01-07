import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from src.additional_loss import AdditionalLossCalculator

path = (
    "/srv/datasets3/HoliCity/images/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_imag.jpg"
)

# 画像の読み込みと前処理
image = Image.open(path)
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # [0,1]の範囲に正規化
    ]
)
image_tensor = transform(image).unsqueeze(0)  # バッチ次元を追加 [1, C, H, W]

# エッジ検出の実行
additional_loss = AdditionalLossCalculator()
edges = additional_loss.detect_edges(image_tensor)  # [1, 2, H, W]
edge_x = edges[0][0]
edge_y = edges[0][1]
# 結果の可視化
edge_magnitude = torch.norm(edges[0], p=2, dim=0)  # エッジの強度を計算
# エッジ強度0.1以下のものは表示しない

plt.figure(figsize=(16, 4))

plt.subplot(141)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(142)
plt.imshow(edge_magnitude.cpu().numpy(), cmap="gray")
plt.title("Edge Map")
plt.axis("off")

plt.subplot(143)
plt.imshow(edge_x.cpu().numpy(), cmap="bwr")
plt.title("Edge X")
plt.axis("off")

plt.subplot(144)
plt.imshow(edge_y.cpu().numpy(), cmap="bwr")
plt.title("Edge Y")
plt.axis("off")

plt.tight_layout()
plt.savefig("edge_map.png")
