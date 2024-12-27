import numpy as np
import cv2


class VanishingPointVisualizer:
    def __init__(self, image_size=(512, 512), angle_step=10):
        self.image_size = image_size
        self.angle_step = angle_step

        # 固定の色定義 (B, G, R, A)形式
        self.colors = {
            "red": [(0, 0, 255, 85), (128, 0, 255, 0)],     # 赤
            "blue": [(255, 0, 0, 85), (255, 0, 128, 0)],    # 青
            "green": [(0, 255, 0, 85), (128, 255, 0, 0)]    # 緑
        }

    def _create_radial_pattern(self, center, colors):
        h, w = self.image_size
        y, x = np.ogrid[:h, :w]

        angles = np.degrees(np.arctan2(y - center[1], x - center[0]))
        angles = (angles + 180) % 360

        pattern = np.zeros((h, w, 4), dtype=np.float32)  # BGRAで4チャンネル

        num_sections = 360 // self.angle_step
        for i in range(num_sections):
            start_angle = i * self.angle_step
            end_angle = (i + 1) * self.angle_step
            color_idx = i % len(colors)

            mask = (angles >= start_angle) & (angles < end_angle)
            pattern[mask] = colors[color_idx]

        return pattern  # BGRAで返す

    def create_condition_image(self, points):
        h, w = self.image_size
        result = np.zeros((h, w, 3), dtype=np.float32)
        
        # インデックスと色の対応を定義（緑→青→赤の順でブレンド）
        color_assignments = [
            (2, "green"),
            (1, "blue"),
            (0, "red")
        ]
        
        # 各色について処理
        for idx, color_key in color_assignments:
            if idx >= len(points):  # 該当するインデックスの消失点が存在しない場合はスキップ
                continue
                
            x, y = points[idx]
            colors = self.colors[color_key]

            # 放射状パターンを生成（BGRAで取得）
            pattern = self._create_radial_pattern((x, y), colors)
            
            # パターンのアルファチャンネルを使って重ね合わせ
            alpha = pattern[:, :, 3:] / 255.0
            result = result * (1 - alpha) + pattern[:, :, :3] * alpha
            
        # 0-255の範囲にクリップして整数に変換
        result = np.clip(result, 0, 255).astype(np.uint8)
        # (0,0,0)のピクセルは(255,255,255)にする
        black_pixels = (result[:, :, 0] == 0) & (result[:, :, 1] == 0) & (result[:, :, 2] == 0)
        result[black_pixels] = [255, 255, 255]

        return result


if __name__ == "__main__":
    visualizer = VanishingPointVisualizer(
        image_size=(512, 512), 
        angle_step=10
    )

    # テスト用の点（3点未満のケースも試す）
    points = [(256, 256), (128, 384), (384, 128)]  # 中心, 左下, 右上
    # points = [(256, 256), (128, 384)]  # 2点の場合
    # points = [(256, 256)]  # 1点の場合
    
    condition_image = visualizer.create_condition_image(points)
    cv2.imwrite("condition_image.png", condition_image)