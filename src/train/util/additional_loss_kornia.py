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
        Perform Canny edge detection on a batch of images.

        Args:
            images (torch.Tensor): Batch image tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: Edge map of shape [B, 1, H, W]
        """
        canny = kornia.filters.Canny(
            low_threshold=low_threshold, high_threshold=high_threshold
        )
        _, x_canny = canny(images)
        return x_canny

    def detect_sobel_edges(self, images, eps=1e-8):
        """
        Perform edge detection using Sobel filter on a batch of images.

        Args:
            images (torch.Tensor): Batch image tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: Edge map (x-direction) of shape [B, C, H, W]
            torch.Tensor: Edge map (y-direction) of shape [B, C, H, W]
            torch.Tensor: Edge magnitude of shape [B, C, H, W]
        """
        # Define 3x3 Sobel filter
        sobel = kornia.filters.SpatialGradient(mode="sobel", order=1, normalized=True)
        edges = sobel(images)  # [B, C, 2, H, W]
        # Unpack the edges
        gx = edges[:, :, 0]
        gy = edges[:, :, 1]

        # Compute gradient magnitude
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
        Predict the original latent representation from the noise prediction and noisy latents.

        Args:
            noise_scheduler: DDPM scheduler
            noisy_latents (torch.Tensor): Noisy latent representation
            model_pred (torch.Tensor): Model's noise prediction
            current_timesteps (torch.Tensor): Current timesteps
            target_timesteps (torch.Tensor): Target timesteps

        Returns:
            torch.Tensor: Predicted original latent representation
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
        Reconstruct the original image from the denoised latents.

        Args:
            vae: AutoencoderKL model
            denoised_latents (torch.Tensor): Denoised latent representation

        Returns:
            torch.Tensor: Predicted original image [B, 3, H, W]
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
        Calculate scores for edges pointing towards vanishing points.

        Args:
            edges: [B, C, 2, H, W]
            magnitudes: [B, C, H, W]
            vanishing_points: [B, 3, 2]
            angle_threshold: Angle threshold (degrees)
            eps: Small value for numerical stability

        Returns:
            [B]: Score for each batch
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
                # Skip if vanishing point is dummy (-1, -1)
                is_valid_vp = ~((vp[0] == -1) & (vp[1] == -1))
                valid_vp_count += is_valid_vp.float()
                # Calculate unit direction vector from each pixel to vanishing point
                vp_dx = vp_x - x_coords
                vp_dy = vp_y - y_coords
                vp_norms = torch.sqrt(vp_dx * vp_dx + vp_dy * vp_dy + eps)
                vp_dx = vp_dx / vp_norms
                vp_dy = vp_dy / vp_norms
                # Calculate cos(θ) using dot product
                cos_theta = torch.abs(dir_x * vp_dx + dir_y * vp_dy)
                # Calculate angle θ (in radians, 0-2π)
                theta = torch.acos(torch.clamp(cos_theta, -0.999999, 0.999999))
                # Convert angle threshold from degrees to radians
                angle_threshold = torch.tensor(
                    angle_threshold * np.pi / 180.0, device=device
                )
                # Calculate edge validity using sigmoid function
                # Smaller angle difference θ gives larger weight
                temperature = 5.0
                valid_edges = 2 * torch.sigmoid(temperature * (angle_threshold - theta))
                # Sum of edge magnitude * valid edges
                edge_weights = magnitude * valid_edges
                batch_total += torch.sum(edge_weights) * is_valid_vp.float()
            # Normalize by the number of valid vanishing points
            result[batch] = batch_total / (valid_vp_count + eps) / H / W
        return result

    def calc_additional_loss(
        self, original_images, pred_images, vanishing_points, timesteps_mask, eps=1e-8
    ):
        """
        Calculate the loss between original and predicted images.
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
        Save images in a batch.

        Args:
            images (torch.Tensor): Image tensor of shape [B, 3, H, W]
            prefix (str): Prefix for the saved file name
        """

        # Create save directory
        save_dir = os.path.join(os.path.dirname(__file__), "..", "generated_images")
        os.makedirs(save_dir, exist_ok=True)

        # Get timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save each image in the batch
        for i in range(images.shape[0]):
            # Convert from [-1, 1] range to [0, 1]
            img = (images[i].clamp(-1, 1) + 1) / 2

            # Generate file name
            filename = f"{prefix}_{timestamp}_batch{i}.png"
            filepath = os.path.join(save_dir, filename)

            # Save the image
            save_image(img, filepath)
            print(f"Saved image to {filepath}")
