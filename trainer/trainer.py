import os
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from contextlib import nullcontext
from hydra.utils import instantiate
from einops import rearrange

from models.build_model import build_model
from trainer.utils import (
    _move_batch_to_device,
    _resolve_autocast_dtype,
    _watch_model_with_wandb,
)
from trainer.utils_latents import (
    _add_latents_to_batch as _add_latents_to_batch_fn,
    _noise_backbone_latents as _noise_backbone_latents_fn,
    _split_context_target as _split_context_target_fn,
)
from trainer.utils_likelihood import (
    _get_gaussian_log_density as _get_gaussian_log_density_fn,
)
from models.models import build_backbone
from utils import load_cfg
from models.models import HeatmapPatchPredictor, HeatmapPixelPredictor
from time import time

# This is for debugging training dataset
from utils.utils_visualisation import (
    _save_decoded_video_comparisons,
    save_batch_and_model_outputs,
)
from utils.utils_format import format_loss_dict
import json


from losses.losses import apply_timestep_weight_to_loss


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


@dataclass
class TrainState:
    epoch: int = 0
    gradient_steps: int = 0


class Trainer:
    _add_latents_to_batch = _add_latents_to_batch_fn
    _split_context_target = _split_context_target_fn
    _noise_backbone_latents = _noise_backbone_latents_fn
    _get_gaussian_log_density = _get_gaussian_log_density_fn

    def __init__(
        self,
        cfg,
        experiment_dirs,
        model=None,
        optimizer=None,
        scheduler=None,
        train_loader=None,
        eval_mini_loader=None,
        eval_loader=None,
        loss_cls=None,
        evaluator=None,
        device=None,
        local_rank=None,
        global_rank=None,
        train_sampler=None,
        wandb_run=None,
        mixed_precision=True,
        args=None,
        transport=None,
        eval_sampler_ode=None,
    ):
        self.cfg = cfg
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_mini_loader = eval_mini_loader
        self.eval_loader = eval_loader
        self.loss_cls = loss_cls
        self.evaluator = evaluator
        self.experiment_dirs = experiment_dirs
        self.device = device
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.train_sampler = train_sampler
        self.train_state = TrainState()
        self.use_mixed_precision = mixed_precision
        self.mixed_precision_dtype = cfg.trainer.training.get(
            "mixed_precision_dtype", "float16"
        )
        self.autocast_dtype = _resolve_autocast_dtype(self.mixed_precision_dtype)
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_mixed_precision and self.autocast_dtype == torch.float16,
        )
        self.logger = logging.getLogger(__name__)
        self.world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        self.logger.info(
            "Using effective batch size %d (per_gpu=%d, world_size=%d)",
            cfg.trainer.training.batch_size_per_gpu * self.world_size,
            cfg.trainer.training.batch_size_per_gpu,
            self.world_size,
        )
        self.profiler = None
        self.wandb_run = wandb_run
        self._wandb_model_watched = False
        self.epoch_running_losses = {}
        # visualizer
        self.log_freq = cfg.trainer.training.log_freq
        # if in debug mode replace log_freq by 1
        if self.args.debug:
            self.log_freq = 1
        self.average_intermediate_layers = cfg.trainer.average_intermediate_layers
        self.extract_single_layer = cfg.trainer.extract_single_layer
        self.use_dinov3_layer_norm = cfg.trainer.use_dinov3_layer_norm
        self.do_final_full_evaluation = cfg.trainer.evaluation.do_final_full_evaluation
        self.patch_size = cfg.model.patch_size

        # Action + proprio conditioning (used by subclasses)
        trainer_cfg = cfg.trainer
        self.action_conditioned = trainer_cfg.get("action_conditioned", False)
        self.proprio_dim = trainer_cfg.get("proprio_dim", 0)

        # load backbone once, keep frozen and in eval
        self.preprocessor, self.backbone, self.backbone_name = build_backbone(cfg.model)
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval().requires_grad_(False)

        # modifications from RAE paper with transport
        self.transport = transport
        self.eval_sampler_ode = eval_sampler_ode
        self.eval_decoder_cfg = cfg.get("evaluation_decoder", None)
        self.evaluation_decoder = None

    def _build_model_kwargs(self, batch):
        """Build kwargs for action + proprio conditioning."""
        kwargs = {}
        if self.action_conditioned and "actions" in batch:
            kwargs["actions"] = batch["actions"]
        if self.proprio_dim > 0 and "states" in batch:
            kwargs["states"] = batch["states"][:, :, : self.proprio_dim]
        return kwargs

    def _autocast_context(self, enabled=None):
        use_autocast = self.use_mixed_precision if enabled is None else enabled
        if not use_autocast:
            return nullcontext()
        return torch.amp.autocast(
            device_type="cuda", dtype=self.autocast_dtype, enabled=True
        )

    def _watch_model(self):
        self._wandb_model_watched = _watch_model_with_wandb(
            self.wandb_run,
            self._wandb_model_watched,
            self.model,
            self.log_freq,
        )

    def _log_step_window(
        self, epoch_idx, num_batches, epoch_start_time, total_batches, loss_dict
    ):
        torch.cuda.synchronize(self.device)
        avg_step_time = (time() - epoch_start_time) / num_batches
        avg_losses = {k: v / self.log_freq for k, v in loss_dict.items()}
        if dist.is_initialized():
            for loss in avg_losses.values():
                dist.all_reduce(loss)
                loss /= dist.get_world_size()
        if self.global_rank == 0:
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {f"step/{k}": v.item() for k, v in avg_losses.items()},
                    step=self.train_state.gradient_steps,
                )
            self.logger.info(
                "Epoch %d step %d/%d | loss_total=%.4f | avg_step_time=%.3fs | breakdown=%s",
                epoch_idx + 1,
                num_batches,
                total_batches,
                avg_losses["loss_total"].item(),
                avg_step_time,
                {k: round(v.item(), 4) for k, v in avg_losses.items()},
            )
        return avg_losses

    def _log_epoch_summary(
        self,
        epoch_idx,
        num_batches,
        epoch_running_losses,
        epoch_start_time,
        include_lr=True,
    ):
        if num_batches == 0:
            return
        epoch_duration = time() - epoch_start_time
        averaged_losses = self._compute_averaged_losses(
            num_batches, epoch_running_losses
        )
        if self.global_rank == 0:
            summary = {"epoch/loss_total": averaged_losses["loss_total"]}
            if include_lr:
                summary["epoch/lr"] = self.optimizer.param_groups[0]["lr"]
            summary["epoch/duration_sec"] = epoch_duration
            if self.wandb_run is not None:
                self.wandb_run.log(summary, step=self.train_state.gradient_steps)
            lr_suffix = (
                f" | lr={self.optimizer.param_groups[0]['lr']:.6f}"
                if include_lr
                else ""
            )
            self.logger.info(
                "[TRAIN] EPOCH %d SUMMARY | loss_total=%.4f%s | duration=%.3fs",
                epoch_idx + 1,
                averaged_losses["loss_total"],
                lr_suffix,
                epoch_duration,
            )

    def save_checkpoint(self):
        """Persist the current model weights for the active epoch."""
        if self.global_rank is not None and self.global_rank != 0:
            return
        checkpoint_dir = self.experiment_dirs["checkpoint_dir"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"epoch_{self.train_state.epoch:04d}.pt"

        model = self.model.module  # Unwrap DDP

        # unwrap torch compile if needed
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        # Prepare checkpoint dictionary
        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "epoch": self.train_state.epoch,
            "gradient_steps": self.train_state.gradient_steps,
        }

        # Add EMA model if it exists
        if getattr(self, "ema_model", None) is not None:
            checkpoint_dict["ema_model_state_dict"] = self.ema_model.state_dict()
            self.logger.info("Including EMA model in checkpoint")

        torch.save(checkpoint_dict, checkpoint_path)
        self.logger.info(
            "Saved checkpoint epoch_%04d.pt to %s",
            self.train_state.epoch,
            checkpoint_dir,
        )

    def accumulate_running_losses(self, target_dict, loss_dict):
        """
        This function adds the losses from loss_dict, to target_dict
        If the loss keys from loss_dict do not exist in target_dict, it creates them
        """
        for loss_name, loss_value in loss_dict.items():
            loss_value = loss_value.detach()
            if loss_name not in target_dict:
                target_dict[loss_name] = torch.zeros(
                    (), device=self.device, dtype=loss_value.dtype
                )
            target_dict[loss_name] += loss_value

    def _log_grad_norm(self, parameters, loss_dict, key):
        grads = [p.grad for p in parameters if p.grad is not None]
        grads_none = [p.grad for p in parameters if p.grad is None]
        norms = [g.detach().float().norm(2) for g in grads]
        loss_dict[key] = torch.linalg.vector_norm(torch.stack(norms), 2)

    def _load_heatmap_heads(self, heatmap_cfg):
        """
        This function loads the heatmap predictors from a config.
        Temporarily used in the run_evaluation functions to load the heatmap predictors.
        """
        cfg = load_cfg(heatmap_cfg.cfg_path).model
        head_specs = {
            "square_patch": (
                HeatmapPatchPredictor,
                cfg.heatmap_patch_predictor,
                "square_heatmap_patch_predictor",
            ),
            "circle_patch": (
                HeatmapPatchPredictor,
                cfg.heatmap_patch_predictor,
                "circle_heatmap_patch_predictor",
            ),
            "square_pixel": (
                HeatmapPixelPredictor,
                cfg.heatmap_pixel_predictor,
                "square_heatmap_pixel_predictor",
            ),
            "circle_pixel": (
                HeatmapPixelPredictor,
                cfg.heatmap_pixel_predictor,
                "circle_heatmap_pixel_predictor",
            ),
        }
        state = torch.load(heatmap_cfg.checkpoint_path, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        heads = {}
        for name, (module_cls, module_cfg, prefix) in head_specs.items():
            module = module_cls(**module_cfg)
            module.load_state_dict(
                {
                    k.split(".", 1)[1]: v
                    for k, v in state.items()
                    if k.startswith(prefix)
                }
            )
            module = module.to(self.device).eval()
            module.requires_grad_(False)
            heads[name] = module
        return heads

    def _load_detector(self, mode):
        from detectron2.config import LazyConfig, instantiate
        from detectron2.checkpoint import DetectionCheckpointer

        # Define the paths
        model_cfg_path = self.det_cfg.model_cfg_path
        checkpoint_path = self.det_cfg.checkpoint_path

        cfg = LazyConfig.load(model_cfg_path)
        model = instantiate(cfg.model)
        if mode == "eval":
            model.eval()
        elif mode == "train":
            model.train()
        else:
            raise ValueError(f"Unknown mode {mode} for detector loading")
        DetectionCheckpointer(model).load(checkpoint_path)
        return model

    def _load_evaluation_decoder(self):
        if self.eval_decoder_cfg is None:
            return None

        decoder_cfg = load_cfg(self.eval_decoder_cfg.cfg_path)
        decoder = build_model(decoder_cfg)
        checkpoint = torch.load(
            self.eval_decoder_cfg.checkpoint_path, map_location="cpu"
        )
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        decoder.load_state_dict(state_dict, strict=True)
        decoder.eval()
        decoder.requires_grad_(False)
        return decoder

    def _decode_evaluation_latent_video(self, latents, reference_video):
        if self.evaluation_decoder is None:
            raise RuntimeError("Evaluation decoder is not configured.")

        batch_size, _, num_frames, _, _ = latents.shape
        flat_latents = rearrange(latents, "b d t h w -> (b t) d h w").contiguous()
        with self._autocast_context():
            decoded = self.evaluation_decoder(flat_latents)["reconstructed_images"]
        decoded = rearrange(
            decoded,
            "(b t) c h w -> b t c h w",
            b=batch_size,
            t=num_frames,
        )
        decoded = decoded.clamp(-1.0, 1.0)
        if reference_video.dtype == torch.uint8:
            return decoded.add(1.0).mul(127.5).round().to(torch.uint8)
        return decoded.add(1.0).mul(0.5).to(reference_video.dtype)

    def _append_heatmaps_to_model_outputs(self, batch, outputs, heads):
        """
        Function calls inference from the heatmap predictors on:
        - context_latents
        - target_latents (used for oracle downstream performance)
        - predicted_latents (used for predicted downstream performance)
        """
        # Latents operate directly in feature space (no normalization); the
        # "*_denormalized" stems are kept as aliases of the raw-latent heatmaps
        # for downstream consumers (visualization).
        name_map = {
            "context_latents": ["context", "context_denormalized"],
            "target_latents": ["oracle", "oracle_denormalized"],
            "predicted_latents": ["predicted"],
        }
        with torch.inference_mode():
            for key, stems in name_map.items():
                latents = batch[key] if key in batch else outputs[key]
                b, _, t, _, _ = latents.shape
                flat = rearrange(latents, "b d t h w -> (b t) d h w")
                if flat.dtype == torch.float16:
                    flat = flat.to(torch.float32)
                sq_patch = heads["square_patch"](flat)  # [B*t, 1, H_patch, W_patch]
                ci_patch = heads["circle_patch"](flat)  # [B*t, 1, H_patch, W_patch]
                sq_pixel = heads["square_pixel"](flat)  # [B*t, 1, H_patch, W_patch]
                ci_pixel = heads["circle_pixel"](flat)  # [B*t, 1, H_patch, W_patch]
                # reshape them in [B, T, num_h_patch, num_w_patch]
                for stem in stems:
                    outputs[f"{stem}_square_patch_heatmap"] = rearrange(
                        sq_patch, "(b t) 1 h w -> b t h w", b=b, t=t
                    )
                    outputs[f"{stem}_circle_patch_heatmap"] = rearrange(
                        ci_patch, "(b t) 1 h w -> b t h w", b=b, t=t
                    )
                    outputs[f"{stem}_square_heatmap_pixel"] = rearrange(
                        sq_pixel, "(b t) 1 h w -> b t h w", b=b, t=t
                    )
                    outputs[f"{stem}_circle_heatmap_pixel"] = rearrange(
                        ci_pixel, "(b t) 1 h w -> b t h w", b=b, t=t
                    )

    def _convert_batch_to_detector_batch_for_backprop(self, batch, model_outputs):
        future_latents = model_outputs[
            "predicted_latents"
        ]  # [B, D, num_target_T, H, W], these are always called "predicted_latents", normalized or not

        B, D, T_tgt, H, W = future_latents.shape
        if T_tgt != self.number_target_latents:
            raise ValueError(
                f"Expected number of target latents {self.number_target_latents}, got {T_tgt}"
            )

        # Flatten latents into (B*T, D, H, W)
        images = rearrange(future_latents, "b d t h w -> (b t) d h w")

        metadata = batch["metadata"]
        video_filenames = metadata["video_filename"]  # list of len B
        image_ids_all = metadata["image_ids"]  # tensor [B, T]
        original_height = metadata["original_height"]  # tensor [B, T]
        original_width = metadata["original_width"]  # tensor [B, T]

        detector_batch = []
        flat_idx = 0

        batch_instances = batch["instances"]  # Nested List [B][T] of Instances objects

        for b_idx in range(B):
            for t_idx in range(T_tgt):
                # Offset metadata indices to point to target frames, not context frames
                metadata_t_idx = self.number_context_latents + t_idx

                instances = batch_instances[b_idx][t_idx]

                detector_batch.append(
                    {
                        "image": images[flat_idx],
                        "height": int(original_height[b_idx, metadata_t_idx]),
                        "width": int(original_width[b_idx, metadata_t_idx]),
                        "image_id": int(image_ids_all[b_idx, metadata_t_idx]),
                        "frame_type": "target",
                        "instances": instances,
                    }
                )
                flat_idx += 1

        return detector_batch

    def _convert_batch_to_detector_batch(self, batch, model_outputs, use_oracle=False):
        context_latents = batch["context_latents"]  # [B, D, T_context, H, W]
        if use_oracle:
            future_latents = batch["target_latents"]  # [B, D, T_target, H, W]
        else:
            future_latents = model_outputs[
                "predicted_latents"
            ]  # [B, D, num_target_T, H, W]
        # concat
        video_latents = torch.cat([context_latents, future_latents], dim=2)
        B, D, T, H, W = video_latents.shape

        if not T == self.number_context_latents + self.number_target_latents:
            raise ValueError(
                f"Expected number of total latents {self.number_context_latents + self.number_target_latents}, got {T}"
            )

        # Flatten latents into (B*T, D, H, W)
        images = rearrange(video_latents, "b d t h w -> (b t) d h w")

        metadata = batch["metadata"]
        video_filenames = metadata["video_filename"]  # list of len B
        image_ids_all = metadata["image_ids"]  # tensor [B, T]
        original_height = metadata["original_height"]  # tensor [B, T]
        original_width = metadata["original_width"]  # tensor [B, T]

        detector_batch = []
        flat_idx = 0

        for b_idx in range(B):
            for t_idx in range(T):
                if t_idx < self.number_context_latents:
                    frame_type = "context"
                else:
                    frame_type = "target"
                detector_batch.append(
                    {
                        "image": images[flat_idx],  # (D, H, W)
                        "height": int(
                            original_height[b_idx, t_idx]
                        ),  # correct indexing
                        "width": int(original_width[b_idx, t_idx]),  # correct indexing
                        "image_id": int(
                            image_ids_all[b_idx, t_idx]
                        ),  # correct indexing
                        "frame_type": frame_type,
                        # "file_name": video_filenames[b_idx],         # if needed per-video
                    }
                )
                flat_idx += 1

        return detector_batch

    def _inference_detector_on_batch(
        self, batch, model_outputs, detector, use_oracle=False
    ):
        from detectron2.evaluation.coco_evaluation import instances_to_coco_json

        # Build detectron2-format input list
        batch_detection = self._convert_batch_to_detector_batch(
            batch, model_outputs, use_oracle=use_oracle
        )

        with torch.inference_mode():
            det_outputs = detector(batch_detection, inference_from_backbone=True)

        coco_results = []
        for inp, out in zip(batch_detection, det_outputs):
            img_id = inp["image_id"]
            frame_type = inp["frame_type"]
            instances = out["instances"]
            instances = instances.to("cpu")
            res_i = instances_to_coco_json(instances, img_id)

            # keep your "+1 category id" logic if your dataset expects it
            for r in res_i:
                r["category_id"] += 1
                r["frame_type"] = frame_type
                r["rank"] = self.global_rank

            coco_results.extend(res_i)

        return coco_results

    def _detach_to_cpu(self, data):
        if isinstance(data, torch.Tensor):
            return data.detach().to("cpu")
        if isinstance(data, dict):
            return {k: self._detach_to_cpu(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            converted = [self._detach_to_cpu(v) for v in data]
            return type(data)(converted) if isinstance(data, tuple) else converted
        return data

    def _setup_eval_dirs(self, final_eval):
        """Creates and returns the necessary directories."""
        dump_dir = self.experiment_dirs["dump_dir"]
        # Distinguish between final evaluation and periodic evaluation
        folder_suffix = "_final" if final_eval else ""
        folder_name = f"epoch_{self.train_state.epoch:03d}{folder_suffix}"

        dump_epoch_dir = os.path.join(dump_dir, folder_name)

        # Only create dirs on rank 0
        if self.global_rank == 0:
            os.makedirs(dump_epoch_dir, exist_ok=True)
            # Create evaluation dir if needed (logic preserved from original)
            if final_eval:
                eval_dir = self.experiment_dirs.get("evaluation_dir")
                if eval_dir:
                    os.makedirs(os.path.join(eval_dir, folder_name), exist_ok=True)

        return dump_epoch_dir

    def _apply_downstream_tasks(
        self, batch, model_outputs, heads, detector, prediction_id=0
    ):
        """Runs Heatmaps and Detectors (Standard + Oracle) and aggregates results."""
        downstream_outputs = {}

        # 1. Heatmaps
        if heads is not None:
            self._append_heatmaps_to_model_outputs(batch, model_outputs, heads)

        # 2. COCO Detection
        if detector is not None:
            # Standard Detection (Predicted Latents)
            coco_batch = self._inference_detector_on_batch(
                batch, model_outputs, detector, use_oracle=False
            )
            model_outputs["coco_detections"] = coco_batch
            preds_list = [p for p in coco_batch if p["frame_type"] == "target"]

            # Inject sample id
            for p in preds_list:
                p["prediction_id"] = prediction_id

            downstream_outputs["coco_predictions"] = preds_list

            # Oracle Detection (Ground Truth Latents)
            # if prediction_id == 0:
            coco_batch_oracle = self._inference_detector_on_batch(
                batch, model_outputs, detector, use_oracle=True
            )
            model_outputs["coco_detections_oracle"] = coco_batch_oracle
            if prediction_id == 0:
                downstream_outputs["coco_predictions_oracle"] = [
                    p for p in coco_batch_oracle if p["frame_type"] == "target"
                ]

        return downstream_outputs

    def _compute_averaged_losses(self, num_batches, epoch_loss_dict):
        """Average and reduce losses across batches and ranks. Returns dict of floats."""
        averaged_losses = {}
        for k, v in epoch_loss_dict.items():
            avg = v / num_batches
            if dist.is_initialized():
                dist.all_reduce(avg)
                avg /= dist.get_world_size()
            averaged_losses[k] = avg.item()
        return averaged_losses

    def _log_final_metrics(
        self,
        eval_results,
        loss_dict,
        eval_start_time,
        final_eval,
        train_eval_metrics=None,
    ):
        """Aggregate loss statistics and downstream metrics, then log them."""
        prefix = "final_eval" if final_eval else "eval"
        summary = loss_dict.copy()
        summary[f"{prefix}/duration_sec"] = time() - eval_start_time

        if eval_results:
            summary.update(eval_results)
        if train_eval_metrics:
            summary.update(train_eval_metrics)
        # Log on main process only
        if self.global_rank == 0:
            if self.wandb_run:
                self.wandb_run.log(summary, step=self.train_state.gradient_steps)

            log_parts = [
                f"[EVAL] epoch {self.train_state.epoch}",
                f"L1 loss={summary[f'{prefix}/l1_latents']:.4f}",
            ]

            if f"{prefix}/coco/ap" in summary:  # Deterministic Predictor
                log_parts.append(f"AP={summary[f'{prefix}/coco/ap']:.4f}")
                log_parts.append(f"AP_medium={summary[f'{prefix}/coco/ap_medium']:.4f}")
                log_parts.append(f"AP_large={summary[f'{prefix}/coco/ap_large']:.4f}")
            elif f"{prefix}/coco/min_ap" in summary:  # Multi-prediction
                log_parts.append(f"Min_AP={summary[f'{prefix}/coco/min_ap']:.4f}")
                log_parts.append(
                    f"Min_AP_medium={summary[f'{prefix}/coco/min_ap_medium']:.4f}"
                )
                log_parts.append(
                    f"Min_AP_large={summary[f'{prefix}/coco/min_ap_large']:.4f}"
                )

                log_parts.append(f"Median_AP={summary[f'{prefix}/coco/median_ap']:.4f}")
                log_parts.append(
                    f"Median_AP_medium={summary[f'{prefix}/coco/median_ap_medium']:.4f}"
                )
                log_parts.append(
                    f"Median_AP_large={summary[f'{prefix}/coco/median_ap_large']:.4f}"
                )

                log_parts.append(f"Max_AP={summary[f'{prefix}/coco/max_ap']:.4f}")
                log_parts.append(
                    f"Max_AP_medium={summary[f'{prefix}/coco/max_ap_medium']:.4f}"
                )
                log_parts.append(
                    f"Max_AP_large={summary[f'{prefix}/coco/max_ap_large']:.4f}"
                )

            if f"{prefix}/coco_oracle/oracle_ap" in summary:
                log_parts.append(
                    f"Oracle_AP={summary[f'{prefix}/coco_oracle/oracle_ap']:.4f}"
                )
                log_parts.append(
                    f"Oracle_AP_medium={summary[f'{prefix}/coco_oracle/oracle_ap_medium']:.4f}"
                )
                log_parts.append(
                    f"Oracle_AP_large={summary[f'{prefix}/coco_oracle/oracle_ap_large']:.4f}"
                )

            log_parts.append(f"duration={summary[f'{prefix}/duration_sec']:.2f}s")

            self.logger.info(" | ".join(log_parts))
            if final_eval:
                # Clean keys for easier parsing (remove 'eval/' prefix and replace '/' with '_')
                clean_summary = {
                    k.replace(f"{prefix}/", "").replace("/", "_"): v
                    for k, v in summary.items()
                }
                headers = ",".join(clean_summary.keys())
                values = ",".join(
                    f"{v:.6f}" if isinstance(v, float) else str(v)
                    for v in clean_summary.values()
                )
                self.logger.info("FINAL_EVAL_HEADERS,%s", headers)
                self.logger.info("FINAL_EVAL_VALUES,%s", values)

    def _save_metrics_json(self):
        "Save accumulated metrics to JSON file (only on rank 0)"
        if self.global_rank == 0:
            with open(self.metrics_json_path, "w") as f:
                json.dump(self.all_metrics, f, indent=2)
            self.logger.info(f"Saved metrics.json to {self.metrics_json_path}")

    def _convert_metrics_to_serializable(self, metrics_dict):
        """Convert tensor values to Python floats for JSON serialization."""
        serializable = {}
        for k, v in metrics_dict.items():
            if hasattr(v, "item"):
                serializable[k] = v.item()
            elif isinstance(v, (int, float, str, bool)):
                serializable[k] = v
            elif isinstance(v, dict):
                serializable[k] = self._convert_metrics_to_serializable(v)
            else:
                serializable[k] = str(v)
        return serializable

    def _check_loss_health(self, loss) -> bool:
        """
        Check if loss is finite and non-NaN
        Args:
            loss: torch.Tensor
        """
        is_nan = torch.isnan(loss).any()
        is_inf = torch.isinf(loss).any()

        if is_nan or is_inf:
            return False
        return True

    def _module_state_str(self, module_name, module):
        if module is None:
            return f"{module_name}=None"
        mode = "train" if module.training else "eval"
        device = next(module.parameters()).device
        return f"{module_name}_mode={mode} {module_name}_device={device}"

    def _detector_state_str(self):
        return self._module_state_str("detector", self.detector)

    def _configure_eval_module_for_phase(
        self, module, phase, module_name, train_in_train_phase
    ):
        if module is None:
            return None
        if phase == "train":
            if train_in_train_phase:
                module.train()
                module = module.to(self.device)
            else:
                module.eval()
                module = module.cpu()
        elif phase == "train_eval":
            module.eval()
            module = module.cpu()
        elif phase == "eval":
            module.eval()
            module = module.to(self.device)
        else:
            raise ValueError(f"Unknown {module_name} phase {phase}")

        self.logger.info(
            "[%s] phase=%s %s",
            module_name.capitalize(),
            phase,
            self._module_state_str(module_name, module),
        )
        return module

    def _configure_detector_for_phase(self, phase):
        self.detector = self._configure_eval_module_for_phase(
            self.detector,
            phase,
            "detector",
            train_in_train_phase=self.detector_loss_enabled,
        )

    def _configure_decoder_for_phase(self, phase):
        if self.eval_decoder_cfg is None:
            return
        if self.evaluation_decoder is None:
            self.evaluation_decoder = self._load_evaluation_decoder()
        self.evaluation_decoder = self._configure_eval_module_for_phase(
            self.evaluation_decoder,
            phase,
            "evaluation_decoder",
            train_in_train_phase=False,
        )


class TrainerHeatmap(Trainer):
    def __init__(
        self,
        cfg,
        experiment_dirs,
        model,
        optimizer,
        scheduler,
        train_loader,
        eval_loader,
        loss_cls,
        evaluator,
        device,
        local_rank,
        global_rank,
        eval_mini_loader=None,
        train_sampler=None,
        wandb_run=None,
        mixed_precision=True,
        args=None,
        transport=None,
        eval_sampler_ode=None,
    ):
        super().__init__(
            cfg,
            experiment_dirs=experiment_dirs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            eval_mini_loader=eval_mini_loader,
            eval_loader=eval_loader,
            loss_cls=loss_cls,
            evaluator=evaluator,
            device=device,
            local_rank=local_rank,
            global_rank=global_rank,
            train_sampler=train_sampler,
            wandb_run=wandb_run,
            mixed_precision=mixed_precision,
            args=args,
            transport=transport,
            eval_sampler_ode=eval_sampler_ode,
        )

        # training arguments
        self.epochs = cfg.trainer.training.epochs
        self.noise_backbone_latents = cfg.trainer.training.noise_backbone_latents
        self.noise_backbone_latents_std = (
            cfg.trainer.training.noise_backbone_latents_std
        )
        # gradient clipping options
        self.gradient_clip_enabled = cfg.trainer.training.gradient_clipping
        self.gradient_clip_max_norm = cfg.trainer.training.gradient_clipping_max_norm

        self.number_context_latents = cfg.trainer.number_context_latents
        self.number_target_latents = cfg.trainer.number_target_latents

        # evaluation arguments
        self.eval_every = cfg.trainer.evaluation.eval_every
        self.eval_first = cfg.trainer.evaluation.eval_first

        # checkpointing arguments
        self.checkpoint_every = cfg.trainer.checkpoint_every

    def _prepare_targets(self, batch):
        """
        This function is used to convert the raw targets to the format of predictions
        Flatten temporal dimension on the batch
        """
        images = batch["video"]
        batch["images"] = rearrange(images, "b t c h w -> (b t) c h w")
        for key in (
            "target_square_heatmap_pixel",
            "target_circle_heatmap_pixel",
            "target_square_heatmap_patch",
            "target_circle_heatmap_patch",
        ):
            tensor = batch[key]
            batch[key] = rearrange(tensor, "b t ... -> (b t) ...")

    def train(self, profiler=None):
        self.profiler = profiler
        if self.eval_first:
            self.run_evaluation()
        self._watch_model()

        for _ in range(self.epochs):
            epoch_idx = self.train_state.epoch
            self.train_sampler.set_epoch(epoch_idx)
            self.train_epoch(epoch_idx)
            self.train_state.epoch += 1
            if (epoch_idx + 1) % self.eval_every == 0:
                self.run_evaluation()
            self.scheduler.step()
            if (epoch_idx + 1) % self.checkpoint_every == 0:
                self.save_checkpoint()
        if self.do_final_full_evaluation:
            self.run_evaluation(final_eval=True)

    def train_epoch(self, epoch_idx):
        self.model.train()
        self.epoch_running_losses = {}
        self.log_losses = {}

        num_batches = 0
        epoch_start_time = time()

        for batch_idx, batch in enumerate(self.train_loader):
            num_batches += 1
            batch = _move_batch_to_device(batch, self.device)
            self._add_latents_to_batch(batch, flatten_temporal_dim=True)
            if self.noise_backbone_latents:
                self._noise_backbone_latents(batch)
            self._prepare_targets(batch)

            loss_dict = self.train_step(batch)
            self.accumulate_running_losses(
                target_dict=self.log_losses, loss_dict=loss_dict
            )  # stores log_dict in log_losses to be logged every_log_freq

            self.train_state.gradient_steps += 1

            # logging
            if num_batches % self.log_freq == 0:
                self._log_step_window(
                    epoch_idx,
                    num_batches,
                    epoch_start_time,
                    len(self.train_loader),
                    self.log_losses,
                )
                self.accumulate_running_losses(
                    self.epoch_running_losses, self.log_losses
                )  # important. At the end of log, accumulate the log_losses in epoch_running_losses
                self.log_losses = {}  # Then resets the log losses for next logging periodj

            if self.profiler is not None:
                self.profiler.step()

        # Final logging of epoch
        if self.log_losses:
            self.accumulate_running_losses(self.epoch_running_losses, self.log_losses)
            self.log_losses = {}
        self._log_epoch_summary(
            epoch_idx,
            num_batches,
            self.epoch_running_losses,
            epoch_start_time,
            include_lr=False,
        )

    def train_step(self, batch):
        self.optimizer.zero_grad()
        with self._autocast_context():
            model_outputs = self.model(batch)
            loss_dict = self.loss_cls.compute_loss(
                batch, model_outputs, compute_downstream_tasks=False
            )

        total_loss = loss_dict["loss_total"]
        self.scaler.scale(total_loss).backward()
        if self.gradient_clip_enabled:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.gradient_clip_max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss_dict

    @torch.inference_mode()
    def run_evaluation(self, final_eval=False):
        self.model.eval()
        assert not self.model.training

        dump_epoch_dir = self._setup_eval_dirs(final_eval)
        self.epoch_running_losses = {}
        num_batches = 0
        num_batches_to_save_for_visu = 2

        for dataloader_batch_idx, batch in enumerate(self.eval_loader):
            batch = _move_batch_to_device(batch, self.device)
            self._add_latents_to_batch(batch, flatten_temporal_dim=True)
            self._prepare_targets(batch)

            with self._autocast_context():
                model_outputs = self.model(batch)
                loss_dict = self.loss_cls.compute_loss(
                    batch, model_outputs, compute_downstream_tasks=True
                )

            self.accumulate_running_losses(self.epoch_running_losses, loss_dict)
            num_batches += 1

            if (
                self.global_rank == 0
                and dataloader_batch_idx <= num_batches_to_save_for_visu
            ):
                save_batch_and_model_outputs(
                    batch=batch,
                    dataloader_batch_idx=dataloader_batch_idx,
                    model_outputs=model_outputs,
                    dump_epoch_dir=dump_epoch_dir,
                    gpu_rank=self.global_rank,
                    prediction_type="heatmaps",
                    split="evaluation",
                    num_context_frames=self.number_context_latents,
                    patch_size=self.patch_size,
                )

        averaged_losses = {}
        for name, value in self.epoch_running_losses.items():
            mean_loss = value / num_batches
            if dist.is_initialized():
                dist.all_reduce(mean_loss)
                mean_loss /= dist.get_world_size()
            averaged_losses[name] = mean_loss.item()

        if self.global_rank == 0:
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {f"eval/{name}": val for name, val in averaged_losses.items()},
                    step=self.train_state.gradient_steps,
                )
            summary = ", ".join(
                f"{name}={val:.4f}" for name, val in averaged_losses.items()
            )
            if final_eval:
                self.logger.info("FINAL evaluation losses: %s", summary)
                headers = ",".join(averaged_losses.keys())
                values = ",".join(f"{val:.6f}" for val in averaged_losses.values())
                self.logger.info("FINAL_EVAL_HEADERS,%s", headers)
                self.logger.info("FINAL_EVAL_VALUES,%s", values)
            else:
                self.logger.info(
                    "Epoch %d evaluation losses: %s",
                    self.train_state.epoch,
                    summary,
                )


class TrainerDeterministic(Trainer):
    def __init__(
        self,
        cfg,
        experiment_dirs,
        model,
        optimizer,
        scheduler,
        train_loader,
        eval_mini_loader,
        eval_loader,
        loss_cls,
        device,
        local_rank,
        global_rank,
        train_sampler=None,
        wandb_run=None,
        mixed_precision=True,
        args=None,
        evaluator=None,
        transport=None,
        eval_sampler_ode=None,
    ):
        super().__init__(
            cfg,
            experiment_dirs=experiment_dirs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            eval_mini_loader=eval_mini_loader,
            eval_loader=eval_loader,
            loss_cls=loss_cls,
            evaluator=evaluator,
            device=device,
            local_rank=local_rank,
            global_rank=global_rank,
            train_sampler=train_sampler,
            wandb_run=wandb_run,
            mixed_precision=mixed_precision,
            args=args,
            transport=transport,
            eval_sampler_ode=eval_sampler_ode,
        )

        trainer_cfg = cfg.trainer
        training_cfg = trainer_cfg.training
        eval_cfg = trainer_cfg.evaluation
        self.det_cfg = cfg.get("evaluation_detection", None)

        self.model_name = cfg.model.model_name

        # training hyper-parameters
        self.epochs = cfg.trainer.training.epochs
        self.eval_every = cfg.trainer.evaluation.eval_every
        self.eval_first = cfg.trainer.evaluation.eval_first

        self.ema_decay = getattr(training_cfg, "ema_decay", None)
        self.ema_model = None

        self.heatmap_cfg = getattr(cfg.trainer.evaluation, "heatmap_predictor", None)
        self.checkpoint_every = cfg.trainer.checkpoint_every
        self.number_context_latents = cfg.trainer.number_context_latents
        self.number_target_latents = cfg.trainer.number_target_latents

        self.detector_loss_cfg = training_cfg.get("detector_loss", None)
        self.detector_loss_enabled = bool(
            self.detector_loss_cfg is not None
            and self.detector_loss_cfg.get("enabled", False)
        )

        downstream_loss = cfg.get("downstream_loss", None)
        if downstream_loss is not None:
            self.loss_downstream_cls = instantiate(downstream_loss)

        # Initialize metrics storage for JSON export
        self.all_metrics = {}
        self.metrics_json_path = os.path.join(
            experiment_dirs["dump_dir"], "metrics.json"
        )

        self.detector = None
        if self.det_cfg is not None:
            try:
                self.detector = self._load_detector(mode="eval")
                # Freeze detector parameters - no updates whatsoever
                for param in self.detector.parameters():
                    param.requires_grad = False
            except (ImportError, ModuleNotFoundError) as exc:
                # Detector-based perception eval is an optional extra that requires
                # detectron2 + detrex (not part of the open-source WM release).
                self.logger.warning(
                    "Detector-based perception eval disabled "
                    "(detectron2/detrex not installed): %s",
                    exc,
                )
                self.detector = None

        # Planning evaluation
        self.n_planning_evals = eval_cfg.get("n_planning_evals", 0)

    def train_step(self, batch):
        self.optimizer.zero_grad()
        with self._autocast_context():
            context_latents = batch["context_latents"]
            model_kwargs = self._build_model_kwargs(batch)
            model_outputs = self.model(context_latents, **model_kwargs)

            loss_dict = self.loss_cls.compute_loss(batch, model_outputs)

        total_loss = loss_dict["loss_total"]
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss_dict

    def train_epoch(self, epoch_idx):
        self.model.train()
        assert self.model.training
        self._configure_detector_for_phase("train")

        self.epoch_running_losses = {}
        self.log_losses = {}
        num_batches = 0
        epoch_start_time = time()
        if self.args.debug:  # If debugging, we log the first batch from the training set as well for each epoch
            saved_train_sample = False

        for batch_idx, batch in enumerate(self.train_loader):
            # If debugging stop after 10 gradient iterations
            if self.args.debug and batch_idx >= 10:
                self.logger.info("Debug mode: Breaking after 10 batches")
                break
            num_batches += 1

            # inspect the batch before move batch to device
            batch = _move_batch_to_device(batch, self.device)
            # inspect the batch after moving batch to device

            self._add_latents_to_batch(batch, flatten_temporal_dim=False)
            self._split_context_target(batch)

            loss_dict = self.train_step(batch)

            self.accumulate_running_losses(
                target_dict=self.log_losses, loss_dict=loss_dict
            )
            self.train_state.gradient_steps += 1

            if num_batches % self.log_freq == 0:
                self._log_step_window(
                    epoch_idx,
                    num_batches,
                    epoch_start_time,
                    len(self.train_loader),
                    self.log_losses,
                )
                self.accumulate_running_losses(
                    target_dict=self.epoch_running_losses, loss_dict=self.log_losses
                )
                self.log_losses = {}

            if self.profiler is not None:
                self.profiler.step()

        if self.log_losses:
            self.accumulate_running_losses(
                target_dict=self.epoch_running_losses, loss_dict=self.log_losses
            )
            self.log_losses = {}

        self._log_epoch_summary(
            epoch_idx,
            num_batches,
            self.epoch_running_losses,
            epoch_start_time,
            include_lr=True,
        )

    @torch.inference_mode()
    def run_evaluation(self, final_eval=False, eval_only_use_mini=False):
        eval_start_time = time()
        self.model.eval()
        assert not self.model.training

        # Setup ressources
        dump_epoch_dir = self._setup_eval_dirs(final_eval)
        num_batches = 0
        num_batches_to_save_for_visu = 5

        heads = (
            self._load_heatmap_heads(self.heatmap_cfg)
            if self.heatmap_cfg is not None
            else None
        )

        self._configure_detector_for_phase("eval")

        self.epoch_running_losses = {}

        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before calling this barrier."
            )
        dist.barrier()

        if final_eval:  # during final eval (or eval_only), we can choose to evaluate on full_val or mini_val
            if eval_only_use_mini:
                eval_loader = self.eval_mini_loader
            else:
                eval_loader = self.eval_loader
        else:  # dyring intermediate evals, we always eval on mini_val
            eval_loader = self.eval_mini_loader

        # 2. Evaluation Loop
        for dataloader_batch_idx, batch in enumerate(eval_loader):
            # Debug mode: only run a few iterations
            num_batches += 1
            if self.args.debug and dataloader_batch_idx >= 3:
                break

            batch = _move_batch_to_device(batch, self.device)
            self._add_latents_to_batch(batch, flatten_temporal_dim=False)
            self._split_context_target(batch)

            with self._autocast_context():
                context_latents = batch["context_latents"]
                model_kwargs = self._build_model_kwargs(batch)
                if self.model_name == "deterministic_latent_predictor":
                    model_outputs = self.model(context_latents, **model_kwargs)
                elif self.model_name == "dino_wm":
                    model_outputs = self.model.module.rollout(
                        context_latents, **model_kwargs
                    )
                else:
                    raise ValueError(
                        f"Unsupported model_name '{self.model_name}' for evaluation"
                    )
                if self.model_name == "deterministic_latent_predictor":
                    loss_dict = self.loss_cls.compute_reconstruction_loss(
                        batch, model_outputs
                    )
                elif self.model_name == "dino_wm":
                    loss_dict = self.loss_cls.compute_rollout_loss(
                        batch, model_outputs
                    )
                # accumulate
                self.accumulate_running_losses(self.epoch_running_losses, loss_dict)

            downstream_outputs = self._apply_downstream_tasks(
                batch, model_outputs, heads, self.detector
            )
            # Accumulate this in the evaluator
            self.evaluator.accumulate(batch, model_outputs, downstream_outputs)

            if (
                self.global_rank == 0
                and dataloader_batch_idx < num_batches_to_save_for_visu
            ):
                save_batch_and_model_outputs(
                    batch=batch,
                    model_outputs=model_outputs,
                    gpu_rank=self.global_rank,
                    prediction_type="latents",
                    split="evaluation",
                    dump_epoch_dir=dump_epoch_dir,
                    dataloader_batch_idx=dataloader_batch_idx,
                    num_context_frames=self.number_context_latents,
                    prediction_id=0,
                    visualize_ground_truths=True,
                    patch_size=self.patch_size,
                )

            if self.global_rank == 0 and num_batches % self.log_freq == 0:
                current_loss = self.epoch_running_losses["l1_latents"] / num_batches
                self.logger.info(
                    "[EVAL] Epoch %d | Batch %d / %d | Current L1 latents loss=%.4f",
                    self.train_state.epoch,
                    dataloader_batch_idx,
                    len(eval_loader),
                    current_loss,
                )

        # 4. METRICS and CLEANUP
        torch.cuda.synchronize()

        prefix = "final_eval/" if final_eval else "eval/"
        self.epoch_running_losses = self._compute_averaged_losses(
            num_batches, self.epoch_running_losses
        )
        self.epoch_running_losses = format_loss_dict(
            self.epoch_running_losses, prefix=prefix
        )

        eval_results = self.evaluator.evaluate(
            logger=self.logger,
            n_predictions_for_evaluation=1,
            list_ids_to_evaluate=getattr(
                eval_loader.dataset, "val_img_ids_list", None
            ),
        )
        eval_results = format_loss_dict(eval_results, prefix=prefix)

        self._log_final_metrics(
            eval_results=eval_results,
            loss_dict=self.epoch_running_losses,
            eval_start_time=eval_start_time,
            final_eval=final_eval,
        )

        # Store metrics for JSON export (only on rank 0)
        if self.global_rank == 0:
            epoch_key = f"epoch_{self.train_state.epoch:04d}" + (
                "_final" if final_eval else ""
            )
            self.all_metrics[epoch_key] = {
                "eval_loss": self.epoch_running_losses,
                "evaluation_metrics": eval_results,
            }
            self._save_metrics_json()

        if heads:
            del heads
        if self.detector is not None:
            if not self.detector_loss_enabled:
                self.detector = self.detector.cpu()
                torch.cuda.empty_cache()

        # Distributed planning evaluation — all ranks participate
        planning_metrics = None
        if self.action_conditioned and self.n_planning_evals > 0:
            planning_metrics = self._run_planning_evaluation()

        # Store metrics for JSON export (only on rank 0)
        if self.global_rank == 0:
            if planning_metrics is not None:
                self.all_metrics[epoch_key]["planning_metrics"] = planning_metrics
                if self.wandb_run:
                    self.wandb_run.log(
                        {f"{prefix}{k}": v for k, v in planning_metrics.items()},
                        step=self.train_state.gradient_steps,
                    )

    @torch.no_grad()
    def _run_planning_evaluation(self):
        raise RuntimeError(
            "PushT action-conditioned planning evaluation is not included in this "
            "release. Keep trainer.evaluation.n_planning_evals=0 (the default)."
        )

    def train(self, profiler=None):
        self.profiler = profiler

        self.model.train()

        if self.eval_first:
            self.run_evaluation(final_eval=False)
        self._watch_model()

        for _ in range(self.epochs):
            epoch_idx = self.train_state.epoch
            self.train_sampler.set_epoch(epoch_idx)

            self.train_epoch(epoch_idx)

            self.train_state.epoch += 1

            self.scheduler.step()

            # Only not in debug mode
            if not self.args.debug:
                if self.train_state.epoch % self.checkpoint_every == 0:
                    self.save_checkpoint()

            if self.args.debug:
                self.run_evaluation(final_eval=False)
            else:
                if self.train_state.epoch % self.eval_every == 0:
                    self.run_evaluation(final_eval=False)

        if self.do_final_full_evaluation:
            self.run_evaluation(final_eval=True)


class TrainerFlowMatching(Trainer):
    """Flow-matching trainer sharing behavior with latent predictor trainer."""

    def __init__(
        self,
        cfg,
        experiment_dirs,
        model,
        optimizer,
        scheduler,
        train_loader,
        eval_mini_loader,
        eval_loader,
        loss_cls,
        device,
        local_rank,
        global_rank,
        train_sampler=None,
        wandb_run=None,
        mixed_precision=True,
        args=None,
        evaluator=None,
        transport=None,
        eval_sampler_ode=None,
    ):

        super().__init__(
            cfg,
            experiment_dirs=experiment_dirs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            eval_mini_loader=eval_mini_loader,
            eval_loader=eval_loader,
            loss_cls=loss_cls,
            device=device,
            local_rank=local_rank,
            global_rank=global_rank,
            train_sampler=train_sampler,
            wandb_run=wandb_run,
            mixed_precision=mixed_precision,
            args=args,
            evaluator=evaluator,
            transport=transport,
            eval_sampler_ode=eval_sampler_ode,
        )

        trainer_cfg = cfg.trainer
        training_cfg = trainer_cfg.training
        eval_cfg = trainer_cfg.evaluation

        # training hyper-parameters
        self.epochs = training_cfg.epochs
        self.gradient_clip_enabled = training_cfg.gradient_clipping
        self.gradient_clip_max_norm = training_cfg.gradient_clipping_max_norm
        self.use_bptt = training_cfg.use_bptt
        self.ema_decay = training_cfg.ema_decay
        self.ema_model = None
        self.model_fn = None
        self.detector_loss_cfg = training_cfg.get("detector_loss", None)
        self.detector_loss_enabled = bool(
            self.detector_loss_cfg is not None
            and self.detector_loss_cfg.get("enabled", False)
        )

        # evaluation settings
        self.eval_every = eval_cfg.eval_every
        self.eval_first = eval_cfg.eval_first
        self.n_predictions_for_evaluation = trainer_cfg.n_predictions_for_evaluation

        # Optional: likelihood configuration during evaluation
        likelihood_cfg = trainer_cfg.likelihood
        self.compute_likelihood = likelihood_cfg.compute_likelihood
        self.exact_divergence = likelihood_cfg.exact_divergence
        self.n_acc = likelihood_cfg.n_acc

        # bookkeeping for checkpoints and latent extraction
        self.checkpoint_every = trainer_cfg.checkpoint_every
        self.number_context_latents = trainer_cfg.number_context_latents
        self.number_target_latents = trainer_cfg.number_target_latents

        # optional evaluation helpers
        self.heatmap_cfg = getattr(cfg.trainer.evaluation, "heatmap_predictor", None)
        self.det_cfg = cfg.get("evaluation_detection", None)

        downstream_loss = cfg.get("downstream_loss", None)
        if downstream_loss is not None:
            self.loss_downstream_cls = instantiate(downstream_loss)

        self.noise_std = float(getattr(self.transport, "noise_std", 1.0))

        # Planning evaluation (only when action-conditioned on PushT)
        self.n_planning_evals = eval_cfg.get("n_planning_evals", 0)
        self.eval_planning_only = eval_cfg.get("eval_planning_only", False)
        self.planning_env = None

        # Initialize metrics storage for JSON export
        self.all_metrics = {}
        self.metrics_json_path = os.path.join(
            experiment_dirs["dump_dir"], "metrics.json"
        )

        self.detector = None
        if self.det_cfg is not None:
            try:
                self.detector = self._load_detector(mode="eval")
                # Freeze detector parameters - no updates whatsoever
                for param in self.detector.parameters():
                    param.requires_grad = False
            except (ImportError, ModuleNotFoundError) as exc:
                # Detector-based perception eval is an optional extra that requires
                # detectron2 + detrex (not part of the open-source WM release).
                self.logger.warning(
                    "Detector-based perception eval disabled "
                    "(detectron2/detrex not installed): %s",
                    exc,
                )
                self.detector = None

    def train_step(self, batch):
        self.optimizer.zero_grad()
        context_latents = batch["context_latents"]
        x = batch["target_latents"]  # shape [B, D, T, H, W]
        model_kwargs = dict(context_latents=context_latents)
        if self.action_conditioned and "actions" in batch:
            model_kwargs["actions"] = batch["actions"]
        if self.proprio_dim > 0 and "states" in batch:
            model_kwargs["states"] = batch["states"][:, :, : self.proprio_dim]
        loss_dict = {}

        # Forward pass through flow model
        with self._autocast_context():
            # loss_tensor = self.transport.training_losses(self.model, x, model_kwargs=model_kwargs)["loss"].mean()
            loss_terms = self.transport.training_losses(
                self.model, x, model_kwargs=model_kwargs
            )
            x_pred = loss_terms["x1_pred"]

        with self._autocast_context():
            # Flow matching loss
            loss_flow_total = loss_terms["loss"].mean()
            if not self._check_loss_health(loss_flow_total):
                self.logger.error(
                    "Rank %s detected unhealthy flow loss (loss_flow_total): %s",
                    self.global_rank,
                    loss_flow_total,
                )

            loss_total = loss_flow_total

        # === Step 3: Detector ===
        if self.detector_loss_enabled and self.detector is not None:
            with self._autocast_context():
                detector_batch = self._convert_batch_to_detector_batch_for_backprop(
                    batch, model_outputs={"predicted_latents": x_pred}
                )
                # Check if x_pred has gradients
                detector_losses = self.detector(
                    detector_batch, inference_from_backbone=True
                )
                detector_loss_total = sum(detector_losses.values())

            if not self._check_loss_health(detector_loss_total):
                self.logger.error(
                    "Rank %s detected unhealthy detector loss (detector_loss_total): %s",
                    self.global_rank,
                    detector_loss_total,
                )

            time_weighting = self.detector_loss_cfg.time_weighting
            detector_time_weight = apply_timestep_weight_to_loss(
                loss_terms["t"],
                weighting=time_weighting,
                gamma=self.detector_loss_cfg.gamma,
            )
            loss_total = loss_total + (
                self.detector_loss_cfg.weight
                * detector_time_weight
                * detector_loss_total
            )
            loss_dict.update({f"detector_{k}": v for k, v in detector_losses.items()})
            loss_dict["detector_loss_total"] = detector_loss_total

        self.scaler.scale(loss_total).backward()

        if self.gradient_clip_enabled:
            self.scaler.unscale_(self.optimizer)
            self._log_grad_norm(
                self.model.parameters(), loss_dict, "flow_grad_norm_pre_clip"
            )
            clip_grad_norm_(self.model.parameters(), self.gradient_clip_max_norm)
            self._log_grad_norm(
                self.model.parameters(), loss_dict, "flow_grad_norm_post_clip"
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        loss_dict["grad_scaler_scale"] = torch.tensor(
            self.scaler.get_scale(), device=self.device
        )

        if self.ema_model is not None and self.ema_decay is not None:
            update_ema(self.ema_model, self.model.module, decay=self.ema_decay)

        # Log losses
        loss_dict["loss_total"] = loss_total
        loss_dict["loss_flow"] = loss_terms["loss_flow"].mean()
        if "loss_consistency" in loss_terms:
            loss_dict["loss_consistency"] = loss_terms["loss_consistency"].mean()
        if "loss_gram" in loss_terms:
            loss_dict["loss_gram"] = loss_terms["loss_gram"].mean()

        return loss_dict

    def train_step_bptt(self, batch):
        """
        BPTT training: unroll full ODE trajectory, compute loss on final/intermediate steps
        """
        context_latents = batch["context_latents"]
        x1_target = batch["target_latents"]
        model_kwargs = dict(context_latents=context_latents)
        loss_dict = {}

        self.optimizer.zero_grad()

        with self._autocast_context():
            x0 = torch.randn_like(x1_target)

            trajectory, log_likelihood = self.eval_sampler_ode(
                x0,
                self.model.forward,
                compute_likelihood=self.compute_likelihood,
                log_p_noise=log_p_noise,
                exact_divergence=self.exact_divergence,
                n_acc=self.n_acc,
                **model_kwargs,
            )

            # trajectory is a list, take the final state
            x1_pred = trajectory[-1]

            # Compute loss between predicted and target
            loss = F.mse_loss(x1_pred, x1_target)

        self.scaler.scale(loss).backward()

        if self.gradient_clip_enabled:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.gradient_clip_max_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.ema_model is not None and self.ema_decay is not None:
            update_ema(self.ema_model, self.model.module, decay=self.ema_decay)

        loss_dict["loss_total"] = loss
        return loss_dict

    def train_epoch(self, epoch_idx):
        epoch_start_time = time()
        # put the model in training mode
        self.model.train()
        assert self.model.training
        self._configure_detector_for_phase("train")

        # initialize variables to monitor for the epoch
        self.epoch_running_losses = {}
        self.log_losses = {}
        num_batches = 0
        if self.args.debug:
            saved_train_sample = False
        train_step_fn = self.train_step_bptt if self.use_bptt else self.train_step

        for batch_idx, batch in enumerate(self.train_loader):
            if self.args.debug and batch_idx >= 3:
                self.logger.info("Debug mode: Breaking after 3 batches")
                break
            num_batches += 1
            batch = _move_batch_to_device(batch, self.device)

            # TODO: Add a normalization of the latents
            self._add_latents_to_batch(batch, flatten_temporal_dim=False)
            self._split_context_target(batch)

            loss_dict = train_step_fn(batch)

            self.accumulate_running_losses(self.log_losses, loss_dict)
            self.train_state.gradient_steps += 1

            # logging
            if num_batches % self.log_freq == 0:
                torch.cuda.synchronize(self.device)
                avg_step_time = (time() - epoch_start_time) / num_batches
                # avg losses is used only for logging, not for accumulation
                avg_losses = {k: v / self.log_freq for k, v in self.log_losses.items()}
                if dist.is_initialized():
                    for k in avg_losses:
                        dist.all_reduce(avg_losses[k])
                        avg_losses[k] /= dist.get_world_size()
                loss_total = avg_losses["loss_total"]
                if self.global_rank == 0 and avg_losses:
                    if self.wandb_run is not None:
                        step_metrics = {
                            f"step/{k}": v.item() for k, v in avg_losses.items()
                        }
                        self.wandb_run.log(
                            step_metrics, step=self.train_state.gradient_steps
                        )
                    self.logger.info(
                        "Epoch %d step %d/%d | Average training loss=%.4f | Average time per gradient step=%.3fs | Loss breakdown: %s",
                        epoch_idx + 1,
                        num_batches,
                        len(self.train_loader),
                        loss_total.item() if loss_total is not None else float("nan"),
                        avg_step_time,
                        {k: round(v.item(), 4) for k, v in avg_losses.items()},
                    )
                self.accumulate_running_losses(
                    self.epoch_running_losses, self.log_losses
                )
                # reset self.log_losses for next gradient update
                self.log_losses = {}

            if self.profiler is not None:
                self.profiler.step()

        if self.log_losses:
            self.accumulate_running_losses(self.epoch_running_losses, self.log_losses)
            self.log_losses = {}

        torch.cuda.synchronize(self.device)
        avg_epoch_losses = {
            k: v / num_batches for k, v in self.epoch_running_losses.items()
        }
        epoch_duration = time() - epoch_start_time

        if dist.is_initialized():
            for k in avg_epoch_losses:
                dist.all_reduce(avg_epoch_losses[k])
                avg_epoch_losses[k] /= dist.get_world_size()

        if self.global_rank == 0:
            epoch_total_time = time() - epoch_start_time
            avg_step_time_epoch = epoch_total_time / num_batches
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.wandb_run is not None:
                epoch_metrics = {
                    f"epoch/{k}": v.item() for k, v in avg_epoch_losses.items()
                }
                epoch_metrics["epoch/avg_step_time"] = avg_step_time_epoch
                epoch_metrics["epoch/lr"] = current_lr
                self.wandb_run.log(epoch_metrics, step=self.train_state.gradient_steps)

            self.logger.info(
                "[TRAIN] EPOCH %d SUMMARY | loss_total=%.4f | lr=%.6f | avg time per gradient step=%.3fs | steps=%d | duration=%.3fs",
                epoch_idx + 1,
                avg_epoch_losses["loss_total"].item(),
                current_lr,
                avg_step_time_epoch,
                num_batches,
                epoch_duration,
            )

    # @torch.inference_mode()
    @torch.no_grad()
    def run_evaluation(self, final_eval=False, eval_only_use_mini=False):
        eval_start_time = time()
        self.model.eval()
        assert not self.model.training
        tracked_model = self.model.module

        # Initialize model_fn if not already set (for eval_first case)
        if self.model_fn is None:
            if self.ema_decay is not None:
                self.ema_model = deepcopy(tracked_model).to(self.device)
                requires_grad(self.ema_model, False)
                update_ema(self.ema_model, tracked_model, decay=0.0)
                self.ema_model.eval()
                self.model_fn = self.ema_model.forward
            else:
                self.model_fn = tracked_model.forward

        train_eval_metrics = None
        if self.train_loader is not None and not self.eval_planning_only:
            train_eval_metrics = self.run_train_set_evaluation(n_samples=256)
        # Setup ressources
        dump_epoch_dir = self._setup_eval_dirs(final_eval)

        prefix = "final_eval/" if final_eval else "eval/"
        eval_results = {}

        if not self.eval_planning_only:
            num_batches = 0
            num_batches_to_save_for_visu = 5

            heads = (
                self._load_heatmap_heads(self.heatmap_cfg)
                if self.heatmap_cfg is not None
                else None
            )

            self._configure_detector_for_phase("eval")
            self._configure_decoder_for_phase("eval")

            self.epoch_running_losses = {}

            if not dist.is_initialized():
                raise RuntimeError(
                    "torch.distributed must be initialized before calling this barrier."
                )
            dist.barrier()

            if final_eval:  # during final eval (or eval_only), we can choose to evaluate on full_val or mini_val
                if eval_only_use_mini:
                    eval_loader = self.eval_mini_loader
                else:
                    eval_loader = self.eval_loader
            else:  # during intermediate evals, we always eval on mini_val
                eval_loader = self.eval_mini_loader

            # 2. Evaluation Loop
            latency_records = []
            for dataloader_batch_idx, batch in enumerate(eval_loader):
                # Debug mode: only run a few iterations
                if self.args.debug and dataloader_batch_idx >= 3:
                    break

                num_batches += 1

                batch = _move_batch_to_device(batch, self.device)

                # --- Encoder timing ---
                torch.cuda.synchronize()
                enc_start = torch.cuda.Event(enable_timing=True)
                enc_end = torch.cuda.Event(enable_timing=True)
                enc_start.record()
                self._add_latents_to_batch(batch, flatten_temporal_dim=False)
                enc_end.record()
                torch.cuda.synchronize()
                encoder_time_ms = enc_start.elapsed_time(enc_end)

                self._split_context_target(batch)

                batch_per_sample_l1 = []
                batch_losses = {}

                # Instead of calling inference only once, we call it n_predictions_for_evaluation times to get several predictions for each sample
                for prediction_id in range(self.n_predictions_for_evaluation):
                    with self._autocast_context():
                        noise_shape = batch["target_latents"].shape
                        zs = torch.randn(
                            noise_shape, device=self.device
                        )  # [B, D, T, H, W]
                        zs = zs * self.noise_std

                        if self.compute_likelihood:
                            log_p_noise = self._get_gaussian_log_density(
                                shape=noise_shape[1:], device=self.device
                            )
                        else:
                            log_p_noise = None

                        context_latents = batch["context_latents"]
                        sample_model_kwargs = dict(context_latents=context_latents)
                        if self.action_conditioned and "actions" in batch:
                            sample_model_kwargs["actions"] = batch["actions"]
                        if self.proprio_dim > 0 and "states" in batch:
                            sample_model_kwargs["states"] = batch["states"][
                                :, :, : self.proprio_dim
                            ]

                        # --- Denoising timing ---
                        torch.cuda.synchronize()
                        denoise_start = torch.cuda.Event(enable_timing=True)
                        denoise_end = torch.cuda.Event(enable_timing=True)
                        denoise_start.record()
                        samples_trajectory, log_likelihood = self.eval_sampler_ode(
                            zs,
                            self.model_fn,
                            compute_likelihood=self.compute_likelihood,
                            log_p_noise=log_p_noise,
                            exact_divergence=self.exact_divergence,
                            n_acc=self.n_acc,
                            **sample_model_kwargs,
                        )
                        denoise_end.record()
                        torch.cuda.synchronize()
                        denoising_time_ms = denoise_start.elapsed_time(denoise_end)
                        total_prediction_time_ms = encoder_time_ms + denoising_time_ms

                        if self.global_rank == 0:
                            self.logger.info(
                                "[LATENCY] batch=%d pred=%d | encoder=%.1f ms | denoising=%.1f ms | total=%.1f ms",
                                dataloader_batch_idx,
                                prediction_id,
                                encoder_time_ms,
                                denoising_time_ms,
                                total_prediction_time_ms,
                            )
                            latency_records.append(
                                {
                                    "encoder_time_ms": encoder_time_ms,
                                    "denoising_time_ms": denoising_time_ms,
                                    "total_prediction_time_ms": total_prediction_time_ms,
                                }
                            )
                        samples = samples_trajectory[-1]
                        model_outputs = {"predicted_latents": samples}
                        # >>> NEW: Log likelihood statistics <
                        if (
                            self.compute_likelihood
                            and self.global_rank == 0
                            and prediction_id == 0
                        ):
                            likelihood = torch.exp(log_likelihood)
                            total_dim = int(np.prod(noise_shape[1:]))  # D * T * H * W
                            bits_per_dim = -log_likelihood.mean() / (
                                np.log(2) * total_dim
                            )

                            self.logger.info(
                                "[LIKELIHOOD] Batch %d | log_likelihood: mean=%.2f, std=%.2f, min=%.2f, max=%.2f | "
                                "likelihood: mean=%.2e, max=%.2e | bits_per_dim=%.4f",
                                dataloader_batch_idx,
                                log_likelihood.mean().item(),
                                log_likelihood.std().item(),
                                log_likelihood.min().item(),
                                log_likelihood.max().item(),
                                likelihood.mean().item(),
                                likelihood.max().item(),
                                bits_per_dim.item(),
                            )

                        loss_dict = self.loss_cls.compute_reconstruction_loss(
                            batch,
                            model_outputs,
                        )

                        per_sample_l1 = loss_dict.pop(
                            f"{self.loss_cls.monitor_loss_name}_per_sample"
                        )  # [B]
                        batch_per_sample_l1.append(per_sample_l1)
                        # accumulate
                        self.accumulate_running_losses(batch_losses, loss_dict)

                    downstream_outputs = self._apply_downstream_tasks(
                        batch,
                        model_outputs,
                        heads,
                        self.detector,
                        prediction_id=prediction_id,
                    )
                    # Accumulate this in the evaluator
                    self.evaluator.accumulate(batch, model_outputs, downstream_outputs)
                    if (
                        self.global_rank == 0
                        and dataloader_batch_idx < num_batches_to_save_for_visu
                    ):
                        save_batch_and_model_outputs(
                            batch=batch,
                            model_outputs=model_outputs,
                            gpu_rank=self.global_rank,
                            prediction_type="latents",
                            split="evaluation",
                            dump_epoch_dir=dump_epoch_dir,
                            dataloader_batch_idx=dataloader_batch_idx,
                            num_context_frames=self.number_context_latents,
                            prediction_id=prediction_id,
                            visualize_ground_truths=True,
                            patch_size=self.patch_size,
                            filter_background=not self.action_conditioned,
                        )
                        if self.eval_decoder_cfg is not None and bool(
                            getattr(self.eval_decoder_cfg, "save_decoded_videos", False)
                        ):
                            predicted_latents_to_decode = model_outputs[
                                "predicted_latents"
                            ]
                            decoded_oracle_videos = (
                                self._decode_evaluation_latent_video(
                                    batch["target_latents"],
                                    batch["video"],
                                )
                            )
                            decoded_predicted_videos = (
                                self._decode_evaluation_latent_video(
                                    predicted_latents_to_decode,
                                    batch["video"],
                                )
                            )
                            _save_decoded_video_comparisons(
                                batch=batch,
                                decoded_oracle_videos=decoded_oracle_videos,
                                decoded_predicted_videos=decoded_predicted_videos,
                                dataloader_batch_idx=dataloader_batch_idx,
                                gpu_rank=self.global_rank,
                                split="evaluation",
                                dump_epoch_dir=dump_epoch_dir,
                                num_context_frames=self.number_context_latents,
                                prediction_id=prediction_id,
                                fps=int(getattr(self.eval_decoder_cfg, "fps", 10)),
                            )

                # Now accumulate the batch_losses
                for k in batch_losses:
                    batch_losses[k] = (
                        batch_losses[k] / self.n_predictions_for_evaluation
                    )

                self.accumulate_running_losses(self.epoch_running_losses, batch_losses)
                batch_per_sample_l1_stacked = torch.stack(
                    batch_per_sample_l1, dim=1
                )  # [B, n_predictions_for_evaluation]

                # Compute stats across predictions (dim=1) then average across batch (mean)
                l1_best = batch_per_sample_l1_stacked.min(dim=1).values.mean()
                l1_worst = batch_per_sample_l1_stacked.max(dim=1).values.mean()
                l1_median = batch_per_sample_l1_stacked.median(dim=1).values.mean()

                prediction_stats = {
                    "l1_latents_best": l1_best,
                    "l1_latents_worst": l1_worst,
                    "l1_latents_median": l1_median,
                }

                self.accumulate_running_losses(
                    self.epoch_running_losses, prediction_stats
                )

                if self.global_rank == 0 and num_batches % self.log_freq == 0:
                    current_loss = self.epoch_running_losses["l1_latents"] / num_batches
                    self.logger.info(
                        "[EVAL] Epoch %d | Batch %d / %d | Current L1 latents loss=%.4f",
                        self.train_state.epoch,
                        dataloader_batch_idx,
                        len(eval_loader),
                        current_loss,
                    )

            # 4. METRICS and CLEANUP
            torch.cuda.synchronize()

            self.epoch_running_losses = self._compute_averaged_losses(
                num_batches, self.epoch_running_losses
            )
            self.epoch_running_losses = format_loss_dict(
                self.epoch_running_losses, prefix=prefix
            )

            eval_results = self.evaluator.evaluate(
                logger=self.logger,
                n_predictions_for_evaluation=self.n_predictions_for_evaluation,
                list_ids_to_evaluate=eval_loader.dataset.val_img_ids_list,
            )
            eval_results = format_loss_dict(eval_results, prefix=prefix)

            self._log_final_metrics(
                eval_results=eval_results,
                loss_dict=self.epoch_running_losses,
                eval_start_time=eval_start_time,
                final_eval=final_eval,
                train_eval_metrics=train_eval_metrics,
            )

            if heads:
                del heads
            if self.detector is not None:
                if not self.detector_loss_enabled:
                    self.detector = self.detector.cpu()
                    torch.cuda.empty_cache()
            if self.evaluation_decoder is not None:
                self._configure_decoder_for_phase("train")

        # Distributed planning evaluation — all ranks participate
        planning_metrics = None
        if self.action_conditioned and self.n_planning_evals > 0:
            planning_metrics = self._run_planning_evaluation()

        # Store metrics for JSON export (only on rank 0)
        if self.global_rank == 0:
            epoch_key = f"epoch_{self.train_state.epoch:04d}" + (
                "_final" if final_eval else ""
            )
            self.all_metrics[epoch_key] = {
                "train_eval_loss": train_eval_metrics,
                "eval_loss": self.epoch_running_losses
                if not self.eval_planning_only
                else {},
                "evaluation_metrics": eval_results,
            }

            if planning_metrics is not None:
                self.all_metrics[epoch_key]["planning_metrics"] = planning_metrics
                if self.wandb_run:
                    self.wandb_run.log(
                        {f"{prefix}{k}": v for k, v in planning_metrics.items()},
                        step=self.train_state.gradient_steps,
                    )

            self._save_metrics_json()

    @torch.inference_mode()
    def run_train_set_evaluation(self, n_samples: int = 1024):
        """
        Run full reconstruction evaluation on a subset of the training set.
        Uses ODE integration (same as eval) rather than single-step prediction.
        """
        eval_start_time = time()
        self.model.eval()
        self._configure_detector_for_phase("train_eval")

        if self.model_fn is None:
            tracked_model = self.model.module
            if self.ema_decay is not None:
                self.ema_model = deepcopy(tracked_model).to(self.device)
                requires_grad(self.ema_model, False)
                update_ema(self.ema_model, tracked_model, decay=0.0)
                self.ema_model.eval()
                self.model_fn = self.ema_model.forward
            else:
                self.model_fn = tracked_model.forward

        # Compute how many batches we need (accounting for distributed training)
        batch_size_per_gpu = self.train_loader.batch_size
        effective_batch_size = batch_size_per_gpu * self.world_size
        n_batches = (n_samples + effective_batch_size - 1) // effective_batch_size

        if self.args.debug:
            n_batches = 3

        self.train_eval_running_losses = {}
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx >= n_batches:
                break

            num_batches += 1
            batch = _move_batch_to_device(batch, self.device)
            self._add_latents_to_batch(batch, flatten_temporal_dim=False)
            self._split_context_target(batch)

            batch_per_sample_l1 = []
            batch_losses = {}

            # for prediction_id in range(self.n_predictions_for_evaluation):
            # Just do one prediction
            for prediction_id in range(1):
                with self._autocast_context():
                    # Sample noise (same logic as run_evaluation)
                    noise_shape = batch["target_latents"].shape
                    zs = torch.randn(noise_shape, device=self.device) * self.noise_std

                    context_latents = batch["context_latents"]
                    sample_model_kwargs = dict(context_latents=context_latents)
                    if self.action_conditioned and "actions" in batch:
                        sample_model_kwargs["actions"] = batch["actions"]
                    sample_trajectory, log_likelihood = self.eval_sampler_ode(
                        zs,
                        self.model_fn,
                        compute_likelihood=False,
                        **sample_model_kwargs,
                    )
                    samples = sample_trajectory[-1]
                    model_outputs = {"predicted_latents": samples}

                    loss_dict = self.loss_cls.compute_reconstruction_loss(
                        batch, model_outputs
                    )
                    per_sample_l1 = loss_dict.pop(
                        f"{self.loss_cls.monitor_loss_name}_per_sample"
                    )
                    batch_per_sample_l1.append(per_sample_l1)

                    self.accumulate_running_losses(batch_losses, loss_dict)

            # Now accumulate the batch_losses
            for k in batch_losses:
                batch_losses[k] = batch_losses[k] / self.n_predictions_for_evaluation

            self.accumulate_running_losses(self.train_eval_running_losses, batch_losses)
            batch_per_sample_l1_stacked = torch.stack(
                batch_per_sample_l1, dim=1
            )  # [B, n_predictions_for_evaluation]

            # Compute stats across predictions (dim=1) then average across batch (mean)
            l1_best = batch_per_sample_l1_stacked.min(dim=1).values.mean()
            l1_worst = batch_per_sample_l1_stacked.max(dim=1).values.mean()
            l1_median = batch_per_sample_l1_stacked.median(dim=1).values.mean()

            prediction_stats = {
                "l1_latents_best": l1_best,
                "l1_latents_worst": l1_worst,
                "l1_latents_median": l1_median,
            }

            self.accumulate_running_losses(
                self.train_eval_running_losses, prediction_stats
            )

        train_eval_metrics = self._compute_averaged_losses(
            num_batches, self.train_eval_running_losses
        )
        train_eval_metrics = format_loss_dict(train_eval_metrics, prefix="train_eval/")
        return train_eval_metrics

    @torch.no_grad()
    def _run_planning_evaluation(self):
        raise RuntimeError(
            "PushT action-conditioned planning evaluation is not included in this "
            "release. Keep trainer.evaluation.n_planning_evals=0 (the default)."
        )

    def train(self, profiler=None):
        self.profiler = profiler
        tracked_model = self.model.module
        if self.ema_decay is not None:
            self.ema_model = deepcopy(tracked_model).to(self.device)
            requires_grad(self.ema_model, False)
            update_ema(self.ema_model, tracked_model, decay=0.0)
            self.ema_model.eval()
            self.model_fn = self.ema_model.forward
        else:
            self.model_fn = tracked_model.forward

        self.model.train()

        if self.eval_first:
            self.run_evaluation(final_eval=False)
        self._watch_model()

        for _ in range(self.epochs):
            epoch_idx = self.train_state.epoch
            self.train_sampler.set_epoch(epoch_idx)

            self.train_epoch(epoch_idx)

            self.train_state.epoch += 1

            self.scheduler.step()

            # Only not in debug mode
            if not self.args.debug:
                if self.train_state.epoch % self.checkpoint_every == 0:
                    self.save_checkpoint()

            if self.args.debug:
                self.run_evaluation(final_eval=False)
            else:
                if self.train_state.epoch % self.eval_every == 0:
                    self.run_evaluation(final_eval=False)

        if self.do_final_full_evaluation:
            self.run_evaluation(final_eval=True)

    def _select_predictions_from_model_outputs(self, model_outputs):
        # select only the keys that are needed for evaluation
        list_of_keys_to_save = [
            # 'predicted_latents',
            "predicted_square_heatmap_pixel",
            "predicted_circle_heatmap_pixel",
        ]
        dict_model_outputs_to_save = {
            key: model_outputs[key] for key in list_of_keys_to_save
        }
        return dict_model_outputs_to_save

    def _write_metadata_in_dict_model_outputs_to_save(
        self, dict_model_outputs_to_save, metadata, prediction_idx
    ):
        # add metadata to the dict to be saved in the evaluator
        dict_model_outputs_to_save["initial_conditions_id"] = metadata[
            "initial_conditions_id"
        ]  # list of str, len [B]
        dict_model_outputs_to_save["trajectory_id"] = metadata[
            "trajectory_id"
        ]  # list of str, len [B]
        dict_model_outputs_to_save["prediction_idx"] = prediction_idx  # int
        return dict_model_outputs_to_save
