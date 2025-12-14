import argparse
import os
from glob import glob

import cv2
import numpy as np
import numpy.linalg as LA
from skimage import measure


FOCAL_LENGTH = 1


def vpt3d_to_2d(w):
    x = w[0] / -w[2] * FOCAL_LENGTH * (512 // 2) + (512 // 2)
    y = -w[1] / -w[2] * FOCAL_LENGTH * (512 // 2) + (512 // 2)
    return x, y


def handle(prefix):
    try:
        print(prefix)
        vpts_path = f"{prefix.replace('/images/', '/vanishing_points/')}_vpts.npz"
        planes_path = f"{prefix.replace('/images/', '/planes/')}_plan.png"
        output_path = f"{prefix.replace('/images/', '/vpts-w-edges/')}_vpts-w-edges.npz"

        # Load vpts and planes
        with np.load(vpts_path) as f:
            vpts = f["vpts"]
            confidence = f["confidence"]
        buf_plane = cv2.imread(planes_path, -1)

        # Extract edge candidates
        lines_candidates = []
        for i in range(np.max(buf_plane)):
            mask_2d = (buf_plane == i + 1).astype(np.float32)
            contours = measure.find_contours(mask_2d, 0.8)
            for contour in contours:
                simplified_contour = measure.approximate_polygon(contour, tolerance=2.0)
                for j in range(len(simplified_contour) - 1):
                    pt1 = np.array(simplified_contour[j][::-1], dtype=np.float32)
                    pt2 = np.array(simplified_contour[j + 1][::-1], dtype=np.float32)
                    lines_candidates.append((pt1, pt2))

        # Detect edges corresponding to each vanishing point
        edges = [[] for _ in range(len(vpts))]
        for idx, vpt in enumerate(vpts):
            vp_coord = np.array(vpt3d_to_2d(vpt))
            for pt1, pt2 in lines_candidates:
                mid = (pt1 + pt2) / 2
                line_length = LA.norm(pt2 - pt1)
                vp_distance = LA.norm(vp_coord - mid)
                if line_length < 1e-6 or vp_distance < 1e-6:
                    continue
                line_dir = (pt2 - pt1) / line_length
                vp_dir = (vp_coord - mid) / vp_distance
                theta = np.arccos(np.clip(abs(np.dot(line_dir, vp_dir)), 0.0, 1.0))
                if theta < 5 * np.pi / 180:
                    edges[idx].append([pt1, pt2])

        # Convert from list to NumPy array
        edges_array = np.array(
            [np.array(group, dtype=np.float32) for group in edges], dtype=object
        )

        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path, vpts=vpts, confidence=confidence, edges=edges_array)
    except Exception as e:
        print(e)


def main():
    import sys
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from src.config import config

    # fmt: off
    parser = argparse.ArgumentParser()
    default_glob = os.path.join(config["dataset"]["image_base_dir"], "*/*.jpg")
    parser.add_argument("--glob", default=default_glob, help="path to the index of pano images")
    parser.add_argument("--batch", type=int, default=0, help="parallel processing")
    parser.add_argument("--total", type=int, default=1, help="parallel procesing")
    # fmt: on
    args = parser.parse_args()

    flist = glob(args.glob)
    flist.sort()
    for f in flist[args.batch :: args.total]:
        handle(f[:-9])


if __name__ == "__main__":
    main()
