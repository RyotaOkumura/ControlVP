from datasets import load_from_disk
from diffusers import AutoencoderKL
import torch
from PIL import Image
import numpy as np

# VAEモデルの読み込み
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
vae = vae.to("cpu")

# データセットの読み込み
dataset_path = "/srv/datasets3/HoliCity/dataset_w_vpts"
dataset = load_from_disk(dataset_path)

data = dataset[16732]
image = data["image"]

# PILイメージをテンソルに変換
# VAEは[-1, 1]の範囲の入力を期待するので、ピクセル値を正規化
image = np.array(image)
image = torch.from_numpy(image).float()
image = image.permute(2, 0, 1).unsqueeze(0)  # [B, C, H, W]の形式に変換
image = (image / 127.5) - 1.0

# VAEでエンコード
with torch.no_grad():
    latents = vae.encode(image).latent_dist.sample()

# 潜在表現を可視化用に処理
# 各チャンネルの平均を取る
latents_vis = latents.mean(dim=1).squeeze()  # [H, W]
# 値を0-255の範囲に正規化
latents_vis = (
    (latents_vis - latents_vis.min()) / (latents_vis.max() - latents_vis.min()) * 255
)
latents_vis = latents_vis.cpu().numpy().astype(np.uint8)

# グレースケール画像として保存
Image.fromarray(latents_vis).save("latents_visualization.png")
