import logging
from typing import Optional

import torch
from einops import rearrange

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_backbone_latents(
    preprocessor,
    backbone,
    backbone_name,
    tensor,
    device,
    flatten_temporal_dim,
    average_intermediate_layers,
    use_dinov3_layer_norm,
    extract_single_layer,
    autocast_dtype=torch.float16,
):
    # assert backbone is in eval
    assert backbone.eval()
    if backbone_name == "dinov3":
        latents, metadata = _extract_dinov3(
            preprocessor,
            backbone,
            tensor,
            device,
            flatten_temporal_dim=flatten_temporal_dim,
            average_intermediate_layers=average_intermediate_layers,
            use_dinov3_layer_norm=use_dinov3_layer_norm,
            extract_single_layer=extract_single_layer,
            autocast_dtype=autocast_dtype,
        )
    else:
        raise ValueError(f"Unsupported backbone '{backbone_name}'")
    return latents, metadata


@torch.no_grad()
def _extract_dinov3(
    preprocessor,
    backbone,
    tensor,
    device,
    flatten_temporal_dim=False,
    average_intermediate_layers=False,
    use_dinov3_layer_norm=False,
    # New parameters for early-layer extraction
    extract_single_layer: Optional[int] = None,  # e.g., 1 for first transformer layer
    autocast_dtype=torch.float16,
):
    """
    input tensor shape: [B, T, C, H, W]
    output latents shape: [B, D, T, H/P, W/P] or [(B T), D, H/P, W/P] if flatten_temporal_dim is True

    If extract_single_layer is specified, extracts that specific layer instead of averaging.
    Layer indices: 0 = patch embeddings, 1-12 = transformer layers
    """
    if average_intermediate_layers and extract_single_layer is not None:
        raise ValueError(
            "average_intermediate_layers and extract_single_layer are mutually exclusive"
        )
    if not average_intermediate_layers and extract_single_layer is None:
        raise ValueError(
            "Set average_intermediate_layers=True or provide extract_single_layer"
        )
    assert tensor.dim() == 5, "Expected backbone input with shape [B, T, C, H, W]"
    b, t, c, h, w = tensor.shape
    tensor = rearrange(tensor, "b t c h w -> (b t) c h w")
    tensor_min = float(tensor.min().item())
    tensor_max = float(tensor.max().item())
    if tensor_min < 0.0 or tensor_max > 255.0:
        raise ValueError(
            f"DINOv3 expects raw image inputs in [0, 255], got min={tensor_min:.3f}, max={tensor_max:.3f}"
        )
    processed = preprocessor(images=tensor.float(), return_tensors="pt")
    pixel_values = processed["pixel_values"].to(device, non_blocking=True)

    with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
        outputs = backbone(pixel_values, output_hidden_states=True)
        hidden_states = (
            outputs.hidden_states
        )  # Tuple of length 13 (first layer is the patch embeddings)
        if average_intermediate_layers:
            list_of_layers_to_get = [3, 6, 9, 12]  # Get layers 3,6,9,12
            list_of_layers_outputs = [
                hidden_states[i][:, 5:, :] for i in list_of_layers_to_get
            ]

            if use_dinov3_layer_norm:
                final_norm = backbone.norm
                list_of_layers_outputs = [
                    final_norm(layer) for layer in list_of_layers_outputs
                ]

            hs = torch.stack(list_of_layers_outputs, dim=-1).mean(
                dim=-1
            )  # [(b t), (num_h_patches * num_w_patches), latent_dim]
        else:
            # Extract specific layer (skip 5 special tokens: CLS + 4 registers)
            hs = hidden_states[extract_single_layer][:, 5:, :]

            logger.info(
                "dinov3 layer %d pre-norm stats | dtype=%s | min/max=%.4f/%.4f | mean/std=%.4f/%.4f",
                extract_single_layer,
                hs.dtype,
                hs.min().item(),
                hs.max().item(),
                hs.float().mean().item(),
                hs.float().std().item(),
            )

            if use_dinov3_layer_norm:
                hs = backbone.norm(hs)

            logger.info(
                "dinov3 layer %d post-norm stats | dtype=%s | min/max=%.4f/%.4f | mean/std=%.4f/%.4f",
                extract_single_layer,
                hs.dtype,
                hs.min().item(),
                hs.max().item(),
                hs.float().mean().item(),
                hs.float().std().item(),
            )

    # at this point hs has shape [(b t), (num_h_patches * num_w_patches), latent_dim]
    num_patches = hs.shape[1]
    patch = int(num_patches**0.5)
    metadata = {
        "batch_size": b,
        "total_time_steps": t,
        "latent_dim": hs.shape[-1],
        "num_h_patches": patch,
        "num_w_patches": patch,
        "extracted_layer": extract_single_layer,
    }
    latents = rearrange(hs, "b (h w) d -> b d h w", h=patch, w=patch)
    if not flatten_temporal_dim:
        latents = rearrange(latents, "(b t) d h w -> b d t h w", b=b, t=t)
    return latents, metadata


def _add_latents_to_batch(self, batch, flatten_temporal_dim):
    """
    This function extracts the latents from the backbone, and add them to the batch.
    The latents are split later in context and targets
    """
    video = batch["video"]  # [B, T, 3, H, W]
    total_needed = self.number_context_latents + self.number_target_latents
    if video.shape[1] != total_needed:
        raise ValueError(f"Expected {total_needed} frames, got {video.shape[1]}")
    video_latents, latents_metadata = extract_backbone_latents(
        self.preprocessor,
        self.backbone,
        self.backbone_name,
        video,
        self.device,
        flatten_temporal_dim,
        average_intermediate_layers=self.average_intermediate_layers,
        use_dinov3_layer_norm=self.use_dinov3_layer_norm,
        extract_single_layer=self.extract_single_layer,
    )

    batch["video_latents"] = video_latents
    batch.update(latents_metadata)


def _split_context_target(self, batch):
    """
    Split the video latents in context_latents, and target_latents.
    Checked in _add_latents_to_batch if context_latents + target_latents = num video time frames
    """
    video_latents = batch["video_latents"]  # shape is [B, D, T, H, W] for videos
    if video_latents.dim() == 5:
        batch["context_latents"] = video_latents[
            :, :, : self.number_context_latents
        ]  # [B, D, num_context_latents, num_h_patches, num_w_patches]
        batch["target_latents"] = video_latents[
            :, :, self.number_context_latents :
        ]  # [B, D, num_target_latents, num_h_patches, num_w_patches]

        batch["num_context_timesteps"] = batch["context_latents"].shape[2]
        batch["num_target_timesteps"] = batch["target_latents"].shape[2]
    else:
        raise ValueError(
            f"Expected `video_latents` to have 5 dimensions, got dim={video_latents.dim()} with shape {tuple(video_latents.shape)}"
        )


def _noise_backbone_latents(self, batch):
    """
    This function noises the video latents.
    It is used in order to make the Heatmap Predictors more robust to imprecise predictions from the Deterministic or Flow predictors
    """
    if self.noise_backbone_latents_std > 0.0:
        batch["video_latents"] = (
            batch["video_latents"]
            + torch.randn_like(batch["video_latents"]) * self.noise_backbone_latents_std
        )
