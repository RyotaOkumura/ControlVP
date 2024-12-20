import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer
from accelerate import Accelerator
import wandb


class ControlNetTrainer:
    def __init__(
        self,
        base_model_path: str,
        condition_model_path: str,
        learning_rate: float = 1e-5,
    ):
        self.accelerator = Accelerator()

        # ControlNetモデルの初期化
        self.controlnet = ControlNetModel.from_pretrained(
            condition_model_path, torch_dtype=torch.float16
        )

        # オプティマイザーの設定
        self.optimizer = torch.optim.AdamW(
            self.controlnet.parameters(), lr=learning_rate
        )

    def train_step(self, batch):
        # 条件画像とターゲット画像を取得
        condition_images = batch["condition_images"]
        target_images = batch["target_images"]

        # ノイズを追加
        noise = torch.randn_like(target_images)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (target_images.shape[0],)
        )

        # ノイズ予測を行う
        with torch.no_grad():
            noisy_images = self.noise_scheduler.add_noise(
                target_images, noise, timesteps
            )

        # ControlNetの予測
        noise_pred = self.controlnet(
            noisy_images, timesteps, condition_images, return_dict=False
        )[0]

        # L2損失の計算
        loss = nn.MSELoss()(noise_pred, noise)

        # 逆伝播
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def train(self, train_dataloader: DataLoader, num_epochs: int = 100):
        # wandbの初期化
        wandb.init(project="controlnet-training")

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_dataloader:
                loss = self.train_step(batch)
                total_loss += loss

            avg_loss = total_loss / len(train_dataloader)
            wandb.log({"train_loss": avg_loss})

            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

            # モデルの保存
            if (epoch + 1) % 10 == 0:
                self.controlnet.save_pretrained(f"controlnet_checkpoint_epoch_{epoch}")
