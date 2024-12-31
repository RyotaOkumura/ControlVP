import torch
import torch.nn.functional as F
import os
from torchvision.utils import save_image
from datetime import datetime


class AdditionalLossCalculator:
    def __init__(
        self,
        distance_from_vanishing_point=10,
    ):
        self.distance_from_vanishing_point = distance_from_vanishing_point

    def print_memory_stats(self, message: str):
        print(f"{message} Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"{message} Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        torch.cuda.empty_cache()

    def detect_edges(self, images):
        """
        バッチ画像からエッジを検出する関数

        Args:
            images (torch.Tensor): shape [B, C, H, W] のバッチ画像テンソル

        Returns:
            torch.Tensor: shape [B, 2, H, W] のエッジマップ
        """
        # 3x3 Sobelフィルタの定義
        sobel_x = (
            torch.tensor(
                [
                    [-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1],
                ],
                dtype=images.dtype,
                device=images.device,
            )
            / 8.0
        )

        sobel_y = (
            torch.tensor(
                [
                    [1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1],
                ],
                dtype=images.dtype,
                device=images.device,
            )
            / 8.0
        )

        # カーネルの形状を調整
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)

        # バッチ内の各画像に対して処理
        B, C, H, W = images.shape

        # グレースケールに変換（RGBの場合）
        if C == 3:
            gray = (
                0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
            )
        else:
            gray = images

        # Sobelフィルタ適用
        edge_x = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), sobel_x)
        edge_y = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), sobel_y)

        # x方向とy方向のエッジを結合
        edges = torch.cat([edge_x, edge_y], dim=1)  # [B, 2, H, W] <- これ正しい
        return edges

    def predict_denoised_latent(
        self,
        noise_scheduler,
        noisy_latents,
        model_pred,
        current_timesteps,
        target_timesteps=None,
    ):
        """
        ノイズ予測値とノイズの乗った潜在表現から、元の潜在表現を予測する関数

        Args:
            noise_scheduler: DDPMスケジューラ
            noisy_latents (torch.Tensor): ノイズの乗った潜在表現
            model_pred (torch.Tensor): モデルによるノイズ予測値
            current_timesteps (torch.Tensor): 現在のタイムステップ
            target_timesteps (torch.Tensor): 目標のタイムステップ

        Returns:
            torch.Tensor: 予測された元の潜在表現
        """
        alpha_prod_t_current = noise_scheduler.alphas_cumprod[current_timesteps].to(
            noisy_latents.device
        )
        alpha_prod_t_current = alpha_prod_t_current.reshape(-1, 1, 1, 1)

        if target_timesteps is not None:
            alpha_prod_t_target = noise_scheduler.alphas_cumprod[target_timesteps].to(
                noisy_latents.device
            )
            alpha_prod_t_target = alpha_prod_t_target.reshape(-1, 1, 1, 1)
            alpha_prod_t = alpha_prod_t_current / alpha_prod_t_target
        else:
            alpha_prod_t = alpha_prod_t_current
        sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
        if noise_scheduler.config.prediction_type == "epsilon":
            # x0 = (x_t - √(1-αt)εθ(x_t)) / √αt
            pred_original_latent = (
                noisy_latents - sqrt_one_minus_alpha_prod_t * model_pred
            ) / sqrt_alpha_prod_t

        elif noise_scheduler.config.prediction_type == "v_prediction":
            raise ValueError("v_prediction is not supported")

        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )
        return pred_original_latent

    def predict_original_image(self, vae, denoised_latents):
        """
        ノイズ予測から元の画像を復元する関数

        Args:
            vae: AutoencoderKL モデル
            denoised_latents (torch.Tensor): ノイズ除去後の潜在表現

        Returns:
            torch.Tensor: 予測された元画像 [B, 3, H, W]
        """

        # VAEでデコード
        # スケーリングファクターで割ってからデコード
        # print(f"denoised_latents: {denoised_latents.shape}")
        # denoised_latentsにNaNがないことを確認
        if torch.isnan(denoised_latents).any():
            print(f"denoised_latents: {denoised_latents}")
            raise ValueError("NaN detected in denoised_latents")
        pred_images = vae.decode(denoised_latents / vae.config.scaling_factor).sample
        return pred_images

    def calc_edge_to_vanishing_point(self, edge_map, vanishing_points):
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
                    (x_valid - vp_x).pow(2)
                    + (y_valid - vp_y).pow(2)
                    - self.distance_from_vanishing_point**2
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
        return result

    def calc_additional_loss(self, original_images, pred_images, vanishing_points):
        """
        元の画像と予測画像の差分を計算する関数
        """
        edge_maps_original = self.detect_edges(original_images)
        edge_maps_pred = self.detect_edges(pred_images)
        original_edge_to_vanishing_point = self.calc_edge_to_vanishing_point(
            edge_maps_original, vanishing_points
        )
        pred_edge_to_vanishing_point = self.calc_edge_to_vanishing_point(
            edge_maps_pred, vanishing_points
        )
        loss = F.mse_loss(
            original_edge_to_vanishing_point, pred_edge_to_vanishing_point
        )
        return loss

    def save_images(self, images, prefix):
        """
        バッチ内の画像を保存する関数

        Args:
            images (torch.Tensor): [B, 3, H, W] の画像テンソル
            prefix (str): 保存するファイル名のプレフィックス
        """

        # 保存ディレクトリの作成
        save_dir = "generated_images"
        os.makedirs(save_dir, exist_ok=True)

        # タイムスタンプの取得
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # バッチ内の各画像を保存
        for i in range(images.shape[0]):
            # [-1, 1]の範囲を[0, 1]に変換
            img = (images[i].clamp(-1, 1) + 1) / 2

            # ファイル名の生成
            filename = f"{prefix}_{timestamp}_batch{i}.png"
            filepath = os.path.join(save_dir, filename)

            # 画像の保存
            save_image(img, filepath)
            print(f"Saved image to {filepath}")
