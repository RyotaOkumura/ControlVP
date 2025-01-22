import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from skimage import measure


FOCAL_LENGTH = 1
IMAGE_SIZE = 512


# 3D座標から2D座標への変換
def vpt3d_to_2d(w):
    x = w[0] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
    y = -w[1] / -w[2] * FOCAL_LENGTH * (IMAGE_SIZE // 2) + (IMAGE_SIZE // 2)
    return x, y


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

    with np.load(
        "/srv/datasets3/HoliCity/vanishing_points/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_vpts.npz"
    ) as f:
        vpts = f["vpts"]

    # draw a nice overlay
    lines_candidates = []
    for i in range(len(plane_normal)):
        mask_2d = (buf_plane == i + 1).astype(np.float32)
        countours = measure.find_contours(mask_2d, 0.8)
        for countour in countours:
            simplified_countour = measure.approximate_polygon(countour, tolerance=2.0)
            for j in range(len(simplified_countour) - 1):
                pt1 = np.array(simplified_countour[j][::-1], dtype=np.float32)
                pt2 = np.array(simplified_countour[j + 1][::-1], dtype=np.float32)
                lines_candidates.append((pt1, pt2))

    edges = [[] for _ in range(len(vpts))]
    for idx, vpt in enumerate(vpts):
        vp_coord = np.array(vpt3d_to_2d(vpt))
        for pt1, pt2 in lines_candidates:
            mid = (pt1 + pt2) / 2
            line_dir = (pt2 - pt1) / LA.norm(pt2 - pt1)
            vp_dir = (vp_coord - mid) / LA.norm(vp_coord - mid)
            theta = np.arccos(np.clip(abs(np.dot(line_dir, vp_dir)), 0.0, 1.0))
            if theta < 5 * np.pi / 180:
                edges[idx].append((pt1, pt2))
    # print(edges)
    for idx, edge in enumerate(edges):
        color = palette[idx]
        for pt1, pt2 in edge:
            cv2.line(
                I, tuple(pt1.astype(int)), tuple(pt2.astype(int)), color, thickness=2
            )
    edges_array = np.array(edges, dtype=object)
    # print(edges_array)
    print(edges_array[0])
    plt.figure(), plt.title("overlay"), plt.imshow(I)
    plt.savefig("plane.png")


if __name__ == "__main__":
    main()
