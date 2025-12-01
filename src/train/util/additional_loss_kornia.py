import torch
import torch.nn.functional as F
import os
from torchvision.utils import save_image
from datetime import datetime
import numpy as np
import kornia


class AdditionalLossCalculatorKornia:
    def print_memory_stats(self, message: str):
        print(f"{message} Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"{message} Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        torch.cuda.empty_cache()

    def detect_canny_edges(self, images, low_threshold=0.5, high_threshold=0.8):
        """
        バッチ画像からキャニーエッジ検出を実行する関数

        Args:
            images (torch.Tensor): shape [B, C, H, W] のバッチ画像テンソル

        Returns:
            torch.Tensor: shape [B, 1, H, W] のエッジマップ
        """
        canny = kornia.filters.Canny(
            low_threshold=low_threshold, high_threshold=high_threshold
        )
        _, x_canny = canny(images)
        return x_canny

    def detect_sobel_edges(self, images, eps=1e-8):
        """
        バッチ画像からソ-ベルフィルタによりエッジ検出を実行する関数

        Args:
            images (torch.Tensor): shape [B, C, H, W] のバッチ画像テンソル

        Returns:
            torch.Tensor: shape [B, C, H, W] のエッジマップ(x方向)
            torch.Tensor: shape [B, C, H, W] のエッジマップ(y方向)
            torch.Tensor: shape [B, C, H, W] のエッジマップの強度
        """
        # 3x3 Sobelフィルタの定義
        sobel = kornia.filters.SpatialGradient(mode="sobel", order=1, normalized=True)
        edges = sobel(images)  # [B, C, 2, H, W]
        # unpack the edges
        gx = edges[:, :, 0]
        gy = edges[:, :, 1]

        # compute gradient maginitude
        magnitude = torch.sqrt(gx * gx + gy * gy + eps)
        return edges, magnitude

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
        alpha_prod_t_current = alpha_prod_t_current.flatten()
        while len(alpha_prod_t_current.shape) < len(noisy_latents.shape):
            alpha_prod_t_current = alpha_prod_t_current.unsqueeze(-1)

        if target_timesteps is not None:
            alpha_prod_t_target = noise_scheduler.alphas_cumprod[target_timesteps].to(
                noisy_latents.device
            )
            alpha_prod_t_target = alpha_prod_t_target.flatten()
            while len(alpha_prod_t_target.shape) < len(noisy_latents.shape):
                alpha_prod_t_target = alpha_prod_t_target.unsqueeze(-1)
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
            pred_original_latent = (
                sqrt_alpha_prod_t * noisy_latents
                - sqrt_one_minus_alpha_prod_t * model_pred
            )
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
        if torch.isnan(denoised_latents).any():
            print(f"denoised_latents: {denoised_latents}")
            raise ValueError("NaN detected in denoised_latents")
        pred_images = vae.decode(denoised_latents / vae.config.scaling_factor).sample
        return pred_images

    def calc_scores(
        self, edges, magnitudes, vanishing_points, angle_threshold=0.0, eps=1e-8
    ):
        """
        エッジマップのうち消失点に向かうものを計算する
        edges: [B, C, 2, H, W]
        magnitudes: [B, C, H, W]
        vanishing_points: [B, 3, 2]
        angle_threshold: 角度閾値（度）
        eps: 小さい値
        returns: [B]
        """
        B, C, _, H, W = edges.shape
        device = edges.device
        result = torch.zeros(B, device=device)
        gx = edges[:, :, 0]
        gy = edges[:, :, 1]
        magnitude = magnitudes
        dir_xs = -gy / magnitude  # [B, C, H, W]
        dir_ys = gx / magnitude  # [B, C, H, W]]
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        for batch in range(B):
            batch_total = 0
            dir_x = dir_xs[batch]
            dir_y = dir_ys[batch]
            vps = vanishing_points[batch]
            valid_vp_count = 0
            for vp in vps:
                vp_x, vp_y = vp[0].item(), vp[1].item()
                # 消失点がダミー(-1, -1)である場合は無効とする
                is_valid_vp = ~((vp[0] == -1) & (vp[1] == -1))
                valid_vp_count += is_valid_vp.float()
                # 各ピクセルから消失点への単位方向ベクトルを計算
                vp_dx = vp_x - x_coords
                vp_dy = vp_y - y_coords
                vp_norms = torch.sqrt(vp_dx * vp_dx + vp_dy * vp_dy + eps)
                vp_dx = vp_dx / vp_norms
                vp_dy = vp_dy / vp_norms
                # 内積によりcosθを計算
                cos_theta = torch.abs(dir_x * vp_dx + dir_y * vp_dy)
                # 角度θを計算（radで0-2pi)
                theta = torch.acos(torch.clamp(cos_theta, -0.999999, 0.999999))
                # 角度閾値を計算（度からラジアンに変換）
                angle_threshold = torch.tensor(
                    angle_threshold * np.pi / 180.0, device=device
                )
                # シグモイド関数によりエッジの有効性を計算。角度差thetaが小さいほど大きな重みを与える
                temperature = 5.0
                valid_edges = 2 * torch.sigmoid(temperature * (angle_threshold - theta))
                # エッジ強度*有効エッジの和をとる
                edge_weights = magnitude * valid_edges
                batch_total += torch.sum(edge_weights) * is_valid_vp.float()
            # 有効な消失点の数で正規化
            result[batch] = batch_total / (valid_vp_count + eps) / H / W
        return result

    def calc_additional_loss(
        self, original_images, pred_images, vanishing_points, timesteps_mask, eps=1e-8
    ):
        """
        元の画像と予測画像の差分を計算する関数
        """
        cannys_original = self.detect_canny_edges(original_images)
        cannys_pred = self.detect_canny_edges(pred_images)
        edges_original, magnitudes_original = self.detect_sobel_edges(
            cannys_original, eps=1e-8
        )
        edges_pred, magnitudes_pred = self.detect_sobel_edges(cannys_pred, eps=1e-8)
        scores_original = self.calc_scores(
            edges_original, magnitudes_original, vanishing_points
        )
        scores_pred = self.calc_scores(edges_pred, magnitudes_pred, vanishing_points)
        masked_scores_original = scores_original * timesteps_mask
        masked_scores_pred = scores_pred * timesteps_mask
        loss = F.mse_loss(
            masked_scores_original, masked_scores_pred, reduction="sum"
        ) / (timesteps_mask.sum() + eps)
        return loss

    def save_images(self, images, prefix):
        """
        バッチ内の画像を保存する関数

        Args:
            images (torch.Tensor): [B, 3, H, W] の画像テンソル
            prefix (str): 保存するファイル名のプレフィックス
        """

        # 保存ディレクトリの作成
        save_dir = os.path.join(os.path.dirname(__file__), "..", "generated_images")
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
