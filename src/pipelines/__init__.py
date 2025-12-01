"""
Custom diffusers pipelines with ControlNet CFG support.

This module provides custom pipeline classes that extend the standard
diffusers ControlNet pipelines with an additional `controlnet_guidance_scale`
parameter for independent classifier-free guidance over ControlNet conditions.
"""

from .controlnet_inpaint_cfg import StableDiffusionControlNetInpaintCFGPipeline

__all__ = [
    "StableDiffusionControlNetInpaintCFGPipeline",
]
