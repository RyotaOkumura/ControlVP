import numpy as np
import matplotlib.pyplot as plt
import cv2

# パス設定
path_vpts = "/srv/datasets3/HoliCity/vanishing_points/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_vpts.npz"
path_img = (
    "/srv/datasets3/HoliCity/images/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_imag.jpg"
)

# データ読み込み
data = np.load(path_vpts)
vpts = data["vpts"]
confidence = data["confidence"]
img = cv2.imread(path_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGRからRGBに変換

# パラメータ
FOCAL_LENGTH = 1
IMAGE_SIZE = 512


def vpt3d_to_2d(w):
    x = w[0] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
    y = -w[1] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
    return x, y


# 画像表示とオーバーレイ
plt.figure(figsize=(12, 12))
plt.imshow(img)

# 各消失点を異なる色でプロット
colors = ["r", "g", "b"]  # 各消失点に異なる色を割り当て
for vpt, conf, color in zip(vpts, confidence, colors):
    x, y = vpt3d_to_2d(vpt)
    print(x, y)
    plt.scatter(x, y, c=color, s=100, label=f"VP (conf={conf})")

    # オプション：消失点から放射状の線を描画
    for t in np.linspace(0, IMAGE_SIZE, 8):
        plt.plot(
            [x, t, x, t], [y, 0, y, IMAGE_SIZE], color=color, alpha=0.3, linestyle="--"
        )

plt.xlim(0, IMAGE_SIZE)
plt.ylim(IMAGE_SIZE, 0)
plt.legend()
plt.title("Vanishing Points Overlay")
plt.savefig("vanishing_points.png")
