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
        if torch.isnan(denoised_latents).any():
            print(f"denoised_latents: {denoised_latents}")
            raise ValueError("NaN detected in denoised_latents")
        pred_images = vae.decode(denoised_latents / vae.config.scaling_factor).sample
        return pred_images

    # def calc_edge_to_vanishing_point(
    #     self, edge_map, vanishing_points, angle_threshold=0.0
    # ):
    #     """
    #     エッジマップのうち消失点に向かうものを計算する
    #     edge_map: [B, 2, H, W]
    #     vanishing_points: [B, 3, 2]
    #     returns: [B]
    #     """
    #     B, _, H, W = edge_map.shape
    #     device = edge_map.device
    #     result = torch.zeros(B, device=device)

    #     edge_magnitudes = self.detect_canny_edges(edge_map)

    #     y_coords, x_coords = torch.meshgrid(
    #         torch.arange(H, device=device),
    #         torch.arange(W, device=device),
    #         indexing="ij",
    #     )

    #     # バッチ内の各サンプルに対して処理
    #     for batch_idx in range(B):
    #         batch_total = 0
    #         y_valid = y_coords
    #         x_valid = x_coords
    #         edge_map_b = edge_map[batch_idx]
    #         edge_magnitude = torch.norm(edge_map_b, p=2, dim=0).clamp(min=1e-8)
    #         edge_dx = edge_map_b[0] / edge_magnitude
    #         edge_dy = edge_map_b[1] / edge_magnitude
    #         dir_x = edge_dy
    #         dir_y = edge_dx
    #         vps = vanishing_points[batch_idx].to(device)  # 現在のデバイスに移動

    #         # print(f"vps: {vps.shape}")
    #         valid_vp_count = 0
    #         for n in range(vps.shape[0]):  # 消失点の数でループ
    #             vp = vps[n]
    #             vp_x, vp_y = vp[0].item(), vp[1].item()  # テンソルから数値に変換
    #             # print(f"vp: {vp_x}, {vp_y}")
    #             if vp_x == -1 and vp_y == -1:
    #                 continue
    #             valid_vp_count += 1

    #             # 消失点への方向ベクトル（正規化）
    #             vp_dy = vp[1] - y_valid
    #             vp_dx = vp[0] - x_valid
    #             vp_norms = torch.norm(torch.stack([vp_dx, vp_dy]), p=2, dim=0).clamp(
    #                 min=1e-8
    #             )
    #             vp_dx = vp_dx / vp_norms
    #             vp_dy = vp_dy / vp_norms

    #             # 方向ベクトル間の内積を計算（cosθ）
    #             cos_theta = torch.abs(dir_x * vp_dx + dir_y * vp_dy)
    #             # θを計算（ラジアン） 0-π
    #             theta = torch.acos(
    #                 torch.clamp(cos_theta, -0.999999, 0.999999)
    #             )  # numerical stabilityのためclamp
    #             # nanが含まれるかチェック
    #             # デバッグ用の値チェック追加
    #             if torch.isnan(theta).any():
    #                 print("Warning: NaN in theta calculation")
    #                 print(
    #                     f"cos_theta range: {cos_theta.min():.6f} to {cos_theta.max():.6f}"
    #                 )
    #                 print(
    #                     f"edge_magnitude range: {edge_magnitude.min():.6f} to {edge_magnitude.max():.6f}"
    #                 )
    #             # 角度の閾値
    #             angle_threshold = torch.tensor(
    #                 angle_threshold * np.pi / 180.0, device=device
    #             )

    #             # temperatureを使用してシグモイド関数を適用
    #             temperature = 5.0
    #             valid_edges = 2 * torch.sigmoid(temperature * (angle_threshold - theta))
    #             # nanが含まれるかチェック
    #             if torch.isnan(valid_edges).any():
    #                 print("valid_edgesにnanが含まれています")
    #                 continue
    #             edge_weights = edge_magnitude * valid_edges
    #             batch_total += torch.sum(edge_weights)
    #         result[batch_idx] = (
    #             batch_total / valid_vp_count / H / W if valid_vp_count > 0 else 0
    #         )
    #     return result

    def calc_scores(
        self, edges, magnitudes, vanishing_points, angle_threshold=0.0, eps=1e-8
    ):
        """
        エッジマップのうち消失点に向かうものを計算する
        edges: [B, C, 2, H, W]
        magnitudes: [B, C, H, W]
        vanishing_points: [B, 3, 2]
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
                is_valid_vp = ~((vp[0] == -1) & (vp[1] == -1))
                valid_vp_count += is_valid_vp.float()
                vp_dx = vp_x - x_coords
                vp_dy = vp_y - y_coords
                vp_norms = torch.sqrt(vp_dx * vp_dx + vp_dy * vp_dy + eps)
                vp_dx = vp_dx / vp_norms
                vp_dy = vp_dy / vp_norms
                cos_theta = torch.abs(dir_x * vp_dx + dir_y * vp_dy)
                theta = torch.acos(torch.clamp(cos_theta, -0.999999, 0.999999))
                angle_threshold = torch.tensor(
                    angle_threshold * np.pi / 180.0, device=device
                )
                temperature = 5.0
                valid_edges = 2 * torch.sigmoid(temperature * (angle_threshold - theta))
                edge_weights = magnitude * valid_edges
                batch_total += torch.sum(edge_weights) * is_valid_vp.float()
            result[batch] = batch_total / (valid_vp_count + eps) / H / W
        return result

    def calc_additional_loss(self, original_images, pred_images, vanishing_points):
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
        loss = F.mse_loss(scores_original, scores_pred)
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
