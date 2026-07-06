from .models import (
    HeatmapLatentPredictor,
    DeterministicLatentPredictor,
    FlowMatchingVideoLatentPredictor,
)
from .dino_wm import ViTPredictor
from .image_decoder import DINOv3ImageDecoder

MODEL_REGISTRY = {
    "heatmap_latent_predictor": HeatmapLatentPredictor,
    "deterministic_latent_predictor": DeterministicLatentPredictor,
    "flow_matching_video_latent_predictor": FlowMatchingVideoLatentPredictor,
    "dino_wm": ViTPredictor,
    "dinov3_image_decoder": DINOv3ImageDecoder,
}


def build_model(config):
    config_full_model = config.model
    model_name = config_full_model.model_name
    model_cls = MODEL_REGISTRY[model_name]
    kwargs = {}
    trainer_cfg = config.get("trainer", None)
    if model_name == "dino_wm":
        dino_cfg = config_full_model.dino_wm
        num_patches = (
            config_full_model.image_size // config_full_model.patch_size
        ) ** 2
        kwargs = {
            "num_patches": num_patches,
            "num_context_latents": dino_cfg.number_context_latents,
            "num_target_latents": trainer_cfg.number_target_latents,
            "input_dim": dino_cfg.input_dim,
            "dim": dino_cfg.dim,
            "depth": dino_cfg.depth,
            "heads": dino_cfg.heads,
            "mlp_dim": dino_cfg.mlp_dim,
            "pool": dino_cfg.pool,
            "dim_head": dino_cfg.dim_head,
            "dropout": dino_cfg.dropout,
            "emb_dropout": dino_cfg.emb_dropout,
            "action_dim": dino_cfg.get("action_dim", 0),
            "proprio_dim": dino_cfg.get("proprio_dim", 0),
            "action_emb_dim": dino_cfg.get("action_emb_dim", 10),
            "proprio_emb_dim": dino_cfg.get("proprio_emb_dim", 10),
            "num_action_repeat": dino_cfg.get("num_action_repeat", 1),
            "num_proprio_repeat": dino_cfg.get("num_proprio_repeat", 1),
        }
        model = model_cls(**kwargs)
        return model
    elif model_name == "deterministic_latent_predictor":
        kwargs = {
            "number_context_latents": trainer_cfg.number_context_latents,
            "number_target_latents": trainer_cfg.number_target_latents,
        }
    elif model_name == "flow_matching_video_latent_predictor":
        kwargs = {
            "number_context_latents": trainer_cfg.number_context_latents,
            "number_target_latents": trainer_cfg.number_target_latents,
        }
    elif model_name == "dinov3_image_decoder":
        kwargs = {}
    model = model_cls(config_full_model, **kwargs)

    return model
