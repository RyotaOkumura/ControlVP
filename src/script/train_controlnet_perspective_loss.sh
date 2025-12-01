uv run accelerate launch \
 src/train/train_controlnet_perspective_loss.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
 --output_dir="ckpts/perspective_loss" \
 --dataset_name=/srv/datasets3/HoliCity/dataset_w_vpts_edges \
 --image_column=image \
 --caption_column=caption \
 --edge_column=edge \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=2 \
 --num_train_epochs=1000 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 --gradient_accumulation_steps=4 \