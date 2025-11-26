# !pip install opencv-python transformers accelerate
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from datasets import load_from_disk
import numpy as np
import torch
from PIL import Image
import cv2
from datetime import datetime
import os


def overlay_vanishing_point(image, vanishing_points):
    """
    Draw radial lines from vanishing points and overlay on image
    image: PIL Image
    vanishing_points: np.array shape=(N, 2) N is the number of vanishing points
    """
    # Convert image to numpy array
    img_array = np.array(image)

    # Get image size
    h, w = img_array.shape[:2]

    # Angle step (every 5 degrees)
    angle_step = 5
    angles = np.arange(0, 360, angle_step)

    # Process each vanishing point
    for vp in vanishing_points:
        # Draw lines for each angle
        for angle in angles:
            theta = np.deg2rad(angle)
            direction = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)

            t = np.linspace(0, max(h, w) * 5, 100)
            points = vp.reshape(2, 1) + direction * t

            # Filter points within image bounds
            mask = (
                (points[0] >= 0) & (points[0] < w) & (points[1] >= 0) & (points[1] < h)
            )
            points = points[:, mask]

            # Draw lines
            if points.shape[1] >= 2:
                for j in range(points.shape[1] - 1):
                    p1 = tuple(map(int, points[:, j]))
                    p2 = tuple(map(int, points[:, j + 1]))
                    cv2.line(img_array, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), 1)

    # Convert numpy array back to PIL Image
    image_with_lines = Image.fromarray(img_array)
    return image_with_lines


class ControlNetWithCFGPipeline:
    def __init__(self, controlnet_model_path, base_model="stabilityai/stable-diffusion-2-base"):
        # Initialize ControlNet and pipeline
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()
    
    def __call__(
        self,
        prompt,
        image,
        height=None,
        width=None,
        num_inference_steps=50,
        timesteps=None,
        sigmas=None,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
        return_dict=True,
        controlnet_conditioning_scale=1.0,
        controlnet_guidance_scale=1.0,
        guess_mode=False,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        **kwargs,
    ):
        # 1. Check inputs (simplified version)
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipe._execution_device
        
        # Set guidance scale
        self.pipe._guidance_scale = guidance_scale
        
        # 2. Define call parameters
        height = height or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        width = width or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        
        # 3. Encode input prompt using library method
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.pipe.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        
        # 4. Prepare image using library method
        control_image = self.pipe.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.controlnet.dtype,
            do_classifier_free_guidance=False,  # We handle CFG ourselves
            guess_mode=guess_mode,
        )
        height, width = control_image.shape[-2:]
        
        # Zero control image for unconditional
        zero_control_image = torch.zeros_like(control_image)
        
        # 5. Prepare timesteps using library method
        from diffusers.pipelines.controlnet.pipeline_controlnet import retrieve_timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        self.pipe._num_timesteps = len(timesteps)
        
        # 6. Prepare latent variables using library method
        num_channels_latents = self.pipe.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        
        # 7. Prepare extra step kwargs
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)
        
        # 7.2 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < control_guidance_start or (i + 1) / len(timesteps) > control_guidance_end)
            ]
            controlnet_keep.append(keeps[0])
        
        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.pipe.scheduler.order
        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.pipe.interrupt:
                    continue
                
                # Expand the latents for ControlNet CFG
                # Order: [uncond_zero, uncond_control, cond_zero, cond_control]
                latent_model_input_expanded = torch.cat([latents] * 4)
                latent_model_input_expanded = self.pipe.scheduler.scale_model_input(latent_model_input_expanded, t)
                
                # Prepare text embeddings
                # Order: [uncond, uncond, cond, cond]
                text_embeddings_expanded = torch.cat([
                    negative_prompt_embeds,  # unconditional
                    negative_prompt_embeds,  # unconditional 
                    prompt_embeds,          # conditional
                    prompt_embeds           # conditional
                ])
                
                # Prepare control images for ControlNet CFG
                # Order: [uncond_zero, uncond_control, cond_zero, cond_control]
                control_images_cfg = torch.cat([
                    zero_control_image,      # unconditional + zero control
                    control_image,           # unconditional + control
                    zero_control_image,      # conditional + zero control
                    control_image            # conditional + control
                ])
                
                # Apply controlnet_keep
                if isinstance(controlnet_keep[i], float):
                    cond_scale = controlnet_conditioning_scale * controlnet_keep[i]
                else:
                    cond_scale = controlnet_conditioning_scale
                
                # ControlNet prediction
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input_expanded,
                    t,
                    encoder_hidden_states=text_embeddings_expanded,
                    controlnet_cond=control_images_cfg,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )
                
                # UNet prediction
                noise_pred = self.pipe.unet(
                    latent_model_input_expanded,
                    t,
                    encoder_hidden_states=text_embeddings_expanded,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.pipe.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
                
                # Custom ControlNet CFG calculation
                # noise_pred = [uncond_zero, uncond_control, cond_zero, cond_control]
                noise_pred_uncond_zero, noise_pred_uncond_control, noise_pred_cond_zero, noise_pred_cond_control = noise_pred.chunk(4)
                
                # ControlNet CFG
                noise_pred_uncond = noise_pred_uncond_zero + controlnet_guidance_scale * (noise_pred_uncond_control - noise_pred_uncond_zero)
                noise_pred_cond = noise_pred_cond_zero + controlnet_guidance_scale * (noise_pred_cond_control - noise_pred_cond_zero)
                
                # Final CFG (same as library)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # Scheduler step (same as library)
                latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                
                # Progress bar update (same as library)
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipe.scheduler.order == 0):
                    progress_bar.update()
        
        # If we do sequential model offloading, let's offload unet and controlnet (same as library)
        if hasattr(self.pipe, "final_offload_hook") and self.pipe.final_offload_hook is not None:
            self.pipe.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()
        
        # Decode (same as library)
        if not output_type == "latent":
            image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image, has_nsfw_concept = self.pipe.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None
        
        # Post-process (same as library)
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        
        image = self.pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        
        # Offload all models (same as library)
        self.pipe.maybe_free_model_hooks()
        
        if not return_dict:
            return (image, has_nsfw_concept)
        
        from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


def main(
    target_image,
    condition_image,
    model_name,
    guidance_scale,
    controlnet_conditioning_scale,
    controlnet_guidance_scale,
    seed,
):
    # Initialize ControlNet CFG pipeline
    pipe_with_cfg = ControlNetWithCFGPipeline(model_name)
    
    # Setup regular Stable Diffusion pipeline (for comparison)
    base_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    base_pipe.scheduler = UniPCMultistepScheduler.from_config(
        base_pipe.scheduler.config
    )
    base_pipe.enable_xformers_memory_efficient_attention()
    base_pipe.enable_model_cpu_offload()

    # Generate images with regular Stable Diffusion
    generator = torch.manual_seed(seed)
    base_images = base_pipe(
        "buildings on both sides of the road, high quality, photorealistic",
        num_inference_steps=20,
        generator=generator,
        num_images_per_prompt=3,
        guidance_scale=guidance_scale,
    ).images

    # Generate images with ControlNet CFG
    generator = torch.manual_seed(seed)
    controlnet_images = pipe_with_cfg(
        "modern buildings, high quality, photorealistic",
        image=condition_image,
        num_inference_steps=20,
        generator=generator,
        num_images_per_prompt=3,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        controlnet_guidance_scale=controlnet_guidance_scale,
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{os.path.dirname(__file__)}/output/controlnet_cfg"
    os.makedirs(output_dir, exist_ok=True)

    # Create grid image list
    num_images = len(base_images)
    grid_images = []

    # First row: target_image and condition_image
    grid_images.extend([target_image, condition_image])

    # Following rows: base_images and controlnet_images
    for base_img, controlnet_img in zip(base_images, controlnet_images):
        grid_images.extend([base_img, controlnet_img])

    # Create grid
    rows = num_images + 1  # target/condition row + generated image rows
    cols = 2
    cell_size = 512  # image size

    # Create empty image
    grid = Image.new("RGB", (cell_size * cols, cell_size * rows))

    # Place images in grid
    for idx, img in enumerate(grid_images):
        row = idx // cols
        col = idx % cols
        grid.paste(img, (col * cell_size, row * cell_size))

    # Save grid image
    grid.save(f"{output_dir}/{timestamp}_cfg_comparison.png")
    print(f"output image saved to {output_dir}/{timestamp}_cfg_comparison.png")


if __name__ == "__main__":
    idx = 20
    model_name = "/home/okumura/lab/grad_thesis_vp/vanishing_point/src/train/model_out_contour_vp_loss_sd2-base_w-10/checkpoint-30000/controlnet"

    dataset_path = "/srv/datasets3/HoliCity/dataset_w_vpts_edges"
    dataset = load_from_disk(dataset_path)
    target_image = dataset[idx]["image"]
    # Create condition image from edges
    edges = dataset[idx]["edge"]
    condition_image = np.zeros((512, 512, 3), dtype=np.uint8)
    if edges:  # If edges exist
        for vp_edges in edges:
            for i in range(0, len(vp_edges), 4):
                x1, y1 = int(vp_edges[i]), int(vp_edges[i + 1])
                x2, y2 = int(vp_edges[i + 2]), int(vp_edges[i + 3])
                cv2.line(
                    condition_image, (x1, y1), (x2, y2), (255, 255, 255), 1
                )  # Draw white lines
    condition_image = Image.fromarray(condition_image)
    
    guidance_scale = 7.5
    controlnet_conditioning_scale = 1.0
    controlnet_guidance_scale = 1.0  # CFG scale for ControlNet
    seed = 4
    
    main(
        target_image,
        condition_image,
        model_name,
        guidance_scale,
        controlnet_conditioning_scale,
        controlnet_guidance_scale,
        seed,
    )