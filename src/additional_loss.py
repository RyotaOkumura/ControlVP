import torch
import torch.nn.functional as F


class AdditionalLossCalculator:
    def __init__(self, distance_from_vanishing_point, edge_magnitude):
        self.distance_from_vanishing_point = distance_from_vanishing_point
        self.edge_magnitude = edge_magnitude

    def detect_edges(self, images):
        """
        バッチ画像からエッジを検出する関数

        Args:
            images (torch.Tensor): shape [B, C, H, W] のバッチ画像テンソル

        Returns:
            torch.Tensor: shape [B, 1, H, W] のエッジマップ
        """
        # Sobelフィルタの定義
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=images.dtype,
            device=images.device,
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=images.dtype,
            device=images.device,
        )

        # カーネルの形状を調整 [1, 1, 3, 3]
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

        # パディング
        pad = F.pad(gray, (1, 1, 1, 1), mode="reflect")

        # 水平・垂直方向のエッジを検出
        edge_x = F.conv2d(pad, sobel_x)
        edge_y = F.conv2d(pad, sobel_y)

        # x方向とy方向のエッジを結合
        edges = torch.cat([edge_x, edge_y], dim=1)  # [B, 2, H, W]

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
        if noise_scheduler.config.prediction_type == "epsilon":
            # εの予測の場合
            alpha_prod_t_current = noise_scheduler.alphas_cumprod[current_timesteps].to(
                noisy_latents.device
            )
            if target_timesteps is not None:
                alpha_prod_t_target = noise_scheduler.alphas_cumprod[
                    target_timesteps
                ].to(noisy_latents.device)
                alpha_prod_t_current = alpha_prod_t_current.reshape(-1, 1, 1, 1)
                alpha_prod_t_target = alpha_prod_t_target.reshape(-1, 1, 1, 1)
                alpha_prod_t = alpha_prod_t_current / alpha_prod_t_target
            else:
                alpha_prod_t = alpha_prod_t_current
            sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
            sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)

            # x0 = (x_t - √(1-αt)εθ(x_t)) / √αt
            pred_original_latent = (
                noisy_latents - sqrt_one_minus_alpha_prod_t * model_pred
            ) / sqrt_alpha_prod_t

        elif noise_scheduler.config.prediction_type == "v_prediction":
            # v予測の場合
            alpha_prod_t_current = noise_scheduler.alphas_cumprod[current_timesteps].to(
                noisy_latents.device
            )
            if target_timesteps is not None:
                alpha_prod_t_target = noise_scheduler.alphas_cumprod[
                    target_timesteps
                ].to(noisy_latents.device)
                alpha_prod_t_current = alpha_prod_t_current.reshape(-1, 1, 1, 1)
                alpha_prod_t_target = alpha_prod_t_target.reshape(-1, 1, 1, 1)
                alpha_prod_t = alpha_prod_t_current / alpha_prod_t_target
            else:
                alpha_prod_t = alpha_prod_t_current

            # x0 = √αt * x_t - √(1-αt) * v
            pred_original_latent = (
                sqrt_alpha_prod_t * noisy_latents
                - sqrt_one_minus_alpha_prod_t * model_pred
            )

        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )

        return pred_original_latent

    def predict_original_image(
        self, vae, noise_scheduler, noisy_latents, model_pred, timesteps
    ):
        """
        ノイズ予測から元の画像を復元する関数

        Args:
            vae: AutoencoderKL モデル
            noise_scheduler: DDPMスケジューラ
            noisy_latents (torch.Tensor): ノイズの乗った潜在表現
            model_pred (torch.Tensor): モデルによるノイズ予測値
            timesteps (torch.Tensor): タイムステップ

        Returns:
            torch.Tensor: 予測された元画像 [B, 3, H, W]
        """
        # まず潜在表現を復元
        pred_original_latent = self.predict_denoised_latent(
            noise_scheduler, noisy_latents, model_pred, timesteps
        )

        # VAEでデコード
        # スケーリングファクターで割ってからデコード
        pred_images = vae.decode(
            pred_original_latent / vae.config.scaling_factor
        ).sample

        return pred_images

    def calc_edge_to_vanishing_point(self, edge_map, vanishing_points):
        """
        エッジマップのうち消失点に向かうものを計算する
        edge_map: [B, 2, H, W]
        vanishing_points: [B, 2, N]
        returns: [B]
        """
        B, _, H, W = edge_map.shape
        N = vanishing_points.shape[-1]
        device = edge_map.device
        result = torch.zeros(B, device=device)

        y_coords, x_coords = torch.meshgrid(
            torch.arrage(H, device=device),
            torch.arrange(W, device=device),
            indexing="ij",
        )

        for b in range(B):
            batch_total = 0
            for n, vp in enumerate(vanishing_points[b]):
                vp_y, vp_x = vp[0], vp[1]

                edge_magnitude = torch.sqrt(torch.sum(edge_map[b] ** 2, dim=0))
                edge_mask = edge_magnitude > self.edge_magnitude

                if not torch.any(edge_mask):
                    continue

                y_valid = y_coords[edge_mask]
                x_valid = x_coords[edge_mask]
                edge_dx = edge_map[b, 1][edge_mask]
                edge_dy = edge_map[b, 0][edge_mask]

                edge_norms = torch.sqrt(edge_dx**2 + edge_dy**2)
                edge_dx = edge_dx / (edge_norms + 1e-6)
                edge_dy = edge_dy / (edge_norms + 1e-6)

                # パラメトリック方程式のパラメータtを計算
                # (x, y) + t * (dx, dy) が消失点からの距離が10以下となるtを求める
                # (x + t*dx - vp_x)^2 + (y + t*dy - vp_y)^2 = 10^2

                a = edge_dx**2 + edge_dy**2  # t^2の係数
                b = 2 * (
                    (x_valid - vp_x) * edge_dx + (y_valid - vp_y) * edge_dy
                )  # tの係数
                c = (
                    (x_valid - vp_x) ** 2
                    + (y_valid - vp_y) ** 2
                    - self.distance_from_vanishing_point**2
                )  # 定数項（半径10の二乗）

                # 判別式
                discriminant = b**2 - 4 * a * c

                # 交点が存在する（判別式が正）場合のみカウント
                valid_edges = discriminant > 0
                batch_total += torch.sum(valid_edges)
            result[b] = batch_total
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
