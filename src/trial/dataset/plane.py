import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from skimage import measure

# palette = sns.color_palette("tab20")
palette = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (1.0, 0.7333333333333333, 0.47058823529411764),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (1.0, 0.596078431372549, 0.5882352941176471),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
    (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),
]


def main():
    prefix = "sample-data/GWUmH4qmANxNTVk_f-_Wrw_HD_060_20"

    I = (
        cv2.imread(
            "/srv/datasets3/HoliCity/images/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_imag.jpg"
        )[:, :, ::-1]
        / 255.0
    )
    buf_plane = cv2.imread(
        "/srv/datasets3/HoliCity/planes/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_plan.png",
        -1,
    )
    with np.load(
        "/srv/datasets3/HoliCity/planes/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_plan.npz"
    ) as f:
        plane_normal = f["ws"]

    # draw a nice overlay
    for i in range(len(plane_normal)):
        mask_2d = (buf_plane == i + 1).astype(np.float32)
        countours = measure.find_contours(mask_2d, 0.8)

        # 3次元のマスクは後で作成
        alpha_fill = mask_2d[..., None]
        edge_mask = np.zeros_like(alpha_fill)
        for countour in countours:
            simplified_countour = measure.approximate_polygon(countour, tolerance=2.0)
            print(simplified_countour)
            for j in range(len(simplified_countour) - 1):
                pt1 = tuple(map(int, simplified_countour[j][::-1]))
                pt2 = tuple(map(int, simplified_countour[j + 1][::-1]))
                cv2.line(edge_mask, pt1, pt2, 1, thickness=2)

        alpha_fill *= 0.5
        color = palette[random.randrange(len(palette))]
        I = I * (1 - alpha_fill) + alpha_fill * color
        I = I * (1 - edge_mask) + edge_mask * color

    plt.figure(), plt.title("overlay"), plt.imshow(I)
    plt.savefig("plane.png")


if __name__ == "__main__":
    main()
