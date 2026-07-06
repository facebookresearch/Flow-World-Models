from einops import rearrange
import torch
import torch.nn.functional as F


def compute_timestep_weights(t, weighting: str, gamma: float = 1.0):
    if weighting == "none":
        return torch.ones_like(t)
    if weighting == "linear":
        return t**gamma
    if weighting == "squared":
        return t**2
    if weighting == "inverse_linear":
        return (1.0 - t) ** gamma
    raise ValueError(
        f"time weighting must be one of 'none', 'linear', 'squared', 'inverse_linear', got '{weighting}'"
    )


def apply_timestep_weight_to_loss(
    t, weighting: str = "none", gamma: float = 1.0, reduction: str = "mean"
):
    weights = compute_timestep_weights(t, weighting, gamma=gamma)
    if reduction == "mean":
        return weights.mean()
    elif reduction == "sum":
        return weights.sum()
    else:
        raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")


class BaseLoss:
    """Base loss class with common utility methods for reshaping latents."""

    def _reshape_latents(self, tensor):
        """
        Reshape latents from [B, D, T, H, W] to [B, (T*D*H*W)].

        Args:
            tensor: Latent tensor of shape [B, D, T, H, W]

        Returns:
            Reshaped tensor of shape [B, (T*D*H*W)]
        """
        if tensor.dim() != 5:
            raise ValueError(
                f"Expected latent tensor of shape [B, D, T, H, W], got dim {tensor.dim()}"
            )
        return rearrange(tensor, "b d t h w -> b (t d h w)")

    def _reshape_latents_per_timestep(self, tensor):
        """
        Reshape latents from [B, D, T, H, W] to [B, T, (D*H*W)].

        Args:
            tensor: Latent tensor of shape [B, D, T, H, W]

        Returns:
            Reshaped tensor of shape [B, T, (D*H*W)]
        """
        if tensor.dim() != 5:
            raise ValueError(
                f"Expected latent tensor of shape [B, D, T, H, W], got dim {tensor.dim()}"
            )
        return rearrange(tensor, "b d t h w -> b t (d h w)")


class ImageReconstructionLoss:
    def __init__(
        self,
        l1_weight: float = 1.0,
        lpips_weight: float = 0.0,
        lpips_net: str = "alex",
    ):
        self.l1_weight = float(l1_weight)
        self.lpips_weight = float(lpips_weight)
        self.lpips_net = lpips_net
        self.lpips = None
        if self.lpips_weight > 0.0:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError(
                    "LPIPS loss requested but lpips is not installed."
                ) from exc
            self.lpips = lpips.LPIPS(net=lpips_net)
            self.lpips.eval()
            for param in self.lpips.parameters():
                param.requires_grad = False

    def to(self, device):
        if self.lpips is not None:
            self.lpips = self.lpips.to(device)
        return self

    def compute_loss(self, batch, model_outputs):
        targets = batch["image_normalized"]
        predictions = model_outputs["reconstructed_images"]

        l1_value = F.l1_loss(predictions, targets)
        total_loss = self.l1_weight * l1_value
        result = {
            "loss_total": total_loss,
            "l1": l1_value,
        }

        if self.lpips is not None:
            lpips_value = self.lpips(predictions.float(), targets.float()).mean()
            result["lpips"] = lpips_value
            result["loss_total"] = (
                result["loss_total"] + self.lpips_weight * lpips_value
            )

        return result


class HeatmapLoss(BaseLoss):
    def __init__(
        self, heatmap_loss_patch=None, heatmap_loss_pixel=None, downstream_loss_fn=None
    ):
        self.heatmap_loss_patch = heatmap_loss_patch
        self.heatmap_loss_pixel = heatmap_loss_pixel
        self.downstream_loss_fn = (
            downstream_loss_fn  # should be l1 in the majority of cases.
        )

    def combine_losses(self, loss_dict):
        return sum(
            value
            for key, value in loss_dict.items()
            if not key.startswith("downstream_")
        )

    def compute_loss(self, batch, model_outputs, compute_downstream_tasks=False):
        loss_dict = {}
        if self.heatmap_loss_patch is not None:
            losses_patch = self.heatmap_loss_patch.compute_loss(
                batch,
                model_outputs,
            )
            loss_dict.update(losses_patch)

        if self.heatmap_loss_pixel is not None:
            losses_pixel = self.heatmap_loss_pixel.compute_loss(
                batch,
                model_outputs,
                compute_downstream_tasks=compute_downstream_tasks,
                downstream_loss_fn=self.downstream_loss_fn,
            )
            loss_dict.update(losses_pixel)

        loss_dict["loss_total"] = self.combine_losses(loss_dict)
        downstream_losses = [
            value for key, value in loss_dict.items() if key.startswith("downstream_")
        ]
        if downstream_losses:
            # this is returning the average l1 loss per dimension, per frame, per shape, in pixel space.
            loss_dict["downstream_loss_total"] = torch.stack(downstream_losses).sum()
        return loss_dict


class HeatmapLossPatch(BaseLoss):
    def __init__(self, loss_fn, weight_loss):
        self.loss_fn = loss_fn
        self.weight_loss = weight_loss

    def _prepare_inputs_and_predictions(self, batch, model_outputs, shape=None):
        if shape not in {"square", "circle"}:
            raise ValueError(f"Unsupported shape '{shape}' for heatmap loss")

        target_key = f"target_{shape}_heatmap_patch"
        prediction_key = f"{shape}_patch_heatmap"

        # Batch targets [B, H/P, W/P, 1]
        targets = batch[target_key]
        # Model predictions [B, 1, H/P, W/P]
        predictions = model_outputs[prediction_key]
        predictions = rearrange(predictions, "b 1 h w -> b h w 1")
        targets = targets.flatten(start_dim=1).argmax(-1).long()
        predictions = predictions.flatten(start_dim=1)

        return targets, predictions

    def compute_loss(self, batch, model_outputs):
        losses = {}
        for shape in ("square", "circle"):
            targets, predictions = self._prepare_inputs_and_predictions(
                batch, model_outputs, shape=shape
            )
            loss = self.loss_fn(predictions, targets) * self.weight_loss
            losses[f"loss_heatmap_patch_{shape}"] = loss
        return losses


class HeatmapLossPixel(BaseLoss):
    def __init__(self, loss_fn, weight_loss):
        self.loss_fn = loss_fn
        self.weight_loss = weight_loss

    def _prepare_inputs_and_predictions(self, batch, model_outputs, shape=None):
        if shape not in {"square", "circle"}:
            raise ValueError(f"Unsupported shape '{shape}' for heatmap loss")

        target_key = f"target_{shape}_heatmap_pixel"
        prediction_key = f"{shape}_heatmap_pixel"

        targets = batch[target_key]
        predictions = model_outputs[prediction_key]

        predictions = rearrange(predictions, "b 1 h w -> b h w 1")
        targets = targets.flatten(start_dim=1).argmax(-1).long()
        predictions = predictions.flatten(start_dim=1)

        return targets, predictions

    def _prepare_inputs_and_predictions_for_downstream_task(
        self, batch, model_outputs, shape=None
    ):
        if shape not in {"square", "circle"}:
            raise ValueError(f"Unsupported shape '{shape}' for heatmap loss")
        target_key = f"target_{shape}_heatmap_pixel"
        prediction_key = f"{shape}_heatmap_pixel"

        targets = batch[target_key].squeeze(-1)  # [B, H, W]
        predictions = model_outputs[prediction_key].squeeze(1)  # [B, H, W]

        flat_predictions = predictions.flatten(start_dim=1)
        max_idx_pred = flat_predictions.argmax(dim=1)
        height, width = predictions.shape[-2:]
        pred_y = max_idx_pred // width
        pred_x = max_idx_pred % width
        predicted_coords = torch.stack((pred_x, pred_y), dim=1)  # [B, 2] -> (x, y)

        flat_targets = targets.flatten(start_dim=1)
        max_idx_target = flat_targets.argmax(dim=1)
        target_y = max_idx_target // width
        target_x = max_idx_target % width
        target_coords = torch.stack((target_x, target_y), dim=1)  # [B, 2] -> (x, y)

        return target_coords, predicted_coords

    def compute_loss(
        self,
        batch,
        model_outputs,
        compute_downstream_tasks=False,
        downstream_loss_fn=None,
    ):
        losses = {}
        for shape in ("square", "circle"):
            targets, predictions = self._prepare_inputs_and_predictions(
                batch, model_outputs, shape=shape
            )
            loss = self.loss_fn(predictions, targets) * self.weight_loss
            losses[f"loss_heatmap_pixel_{shape}"] = loss
            if compute_downstream_tasks:
                if downstream_loss_fn is None:
                    raise ValueError(
                        "downstream_loss_fn must be provided to compute downstream tasks for heatmap supervision"
                    )
                target_coords, predicted_coords = (
                    self._prepare_inputs_and_predictions_for_downstream_task(
                        batch, model_outputs, shape=shape
                    )
                )
                # important note: predicted_coords and target_coords are both in the format (x, y) (which is not the order of access in matrices which is (y, x))
                downstream_loss = downstream_loss_fn(
                    predicted_coords.float(),
                    target_coords.float(),
                )
                losses[f"downstream_loss_heatmap_pixel_{shape}"] = downstream_loss
        return losses


class DeterministicPredictorLoss(BaseLoss):
    def __init__(
        self,
        loss_fn,
        monitor_loss_fn,
        weight_loss: float = 1.0,
        loss_name: str = "loss_total",
        monitor_loss_name: str = "l1_latents",
    ):
        self.loss_fn = loss_fn
        self.monitor_loss_fn = monitor_loss_fn
        self.weight_loss = weight_loss
        self.loss_name = loss_name
        self.monitor_loss_name = (
            monitor_loss_name if monitor_loss_fn is not None else None
        )

    def compute_loss(self, batch, model_outputs):
        predictions = model_outputs["predicted_latents"]
        targets = batch["target_latents"]

        targets = self._reshape_latents(targets)
        predictions = self._reshape_latents(predictions)

        # assert that targets and predictions have the same shape
        if targets.shape != predictions.shape:
            raise ValueError(
                f"Targets and predictions must have the same shape for loss computation, got {targets.shape} and {predictions.shape}"
            )

        loss_value = self.loss_fn(predictions, targets) * self.weight_loss

        # Build the result dictionary
        result = {
            self.loss_name: loss_value,
        }

        return result

    def compute_reconstruction_loss(self, batch, model_outputs):
        targets = batch["target_latents"]
        predictions = model_outputs["predicted_latents"]

        targets = self._reshape_latents(targets)
        predictions = self._reshape_latents(predictions)

        # assert that targets and predictions have the same shape
        if targets.shape != predictions.shape:
            raise ValueError(
                f"Targets and predictions must have the same shape for loss computation, got {targets.shape} and {predictions.shape}"
            )

        loss_value = self.loss_fn(predictions, targets)

        # Build the result dictionary
        result = {
            "l1_latents": loss_value,
        }
        return result


class DinoWMLoss(BaseLoss):
    def __init__(
        self,
        loss_fn,
        monitor_loss_fn,
        weight_loss: float = 1.0,
        loss_name: str = "loss_total",
        monitor_loss_name: str = "l1_latents",
    ):
        self.loss_fn = loss_fn
        self.monitor_loss_fn = monitor_loss_fn
        self.weight_loss = weight_loss
        self.loss_name = loss_name
        self.monitor_loss_name = (
            monitor_loss_name if monitor_loss_fn is not None else None
        )

    def compute_loss(self, batch, model_outputs):
        targets = batch["target_latents"]  # [B, D, T, H, W]
        context = batch["context_latents"]  # [B, D, T, H, W]

        targets = targets[:, :, 0:1, :, :]  # [B, D, 1, H, W]
        context = context[:, :, 1:, :, :]  # [B, D, T-1, H, W]

        # concat on time axis
        targets = torch.cat([context, targets], dim=2)  # [B, D, T, H, W]
        # reshape
        targets = self._reshape_latents(targets)

        predictions = self._reshape_latents(model_outputs["predicted_latents"])

        # assert that targets and predictions have the same shape
        if targets.shape != predictions.shape:
            raise ValueError(
                f"Targets and predictions must have the same shape for loss computation, got {targets.shape} and {predictions.shape}"
            )

        loss_value = self.loss_fn(predictions, targets) * self.weight_loss

        # Build the result dictionary
        result = {
            self.loss_name: loss_value,
        }
        return result

    def compute_rollout_loss(self, batch, model_outputs):
        targets = batch["target_latents"]

        predictions = model_outputs["predicted_latents"]

        # reshape
        targets = self._reshape_latents(targets)

        predictions = self._reshape_latents(predictions)

        # assert that targets and predictions have the same shape
        if targets.shape != predictions.shape:
            raise ValueError(
                f"Targets and predictions must have the same shape for loss computation, got {targets.shape} and {predictions.shape}"
            )

        loss_value = self.loss_fn(predictions, targets)

        # Build the result dictionary
        result = {
            "l1_latents": loss_value,
        }
        return result


class FlowMatchingLoss(BaseLoss):
    """Computes flow-matching loss between predicted vectors and targets."""

    def __init__(
        self,
        loss_fn,
        monitor_loss_fn,
        weight_loss: float = 1.0,
    ):
        self.loss_fn = loss_fn
        self.monitor_loss_fn = monitor_loss_fn
        self.weight_loss = weight_loss
        self.monitor_loss_name = "l1_latents"

    # NOTE: The flow-matching training loss is computed by
    # ``Transport.training_losses`` (see transport/transport.py); this class only
    # provides the L1 latent monitoring loss below.
    def compute_reconstruction_loss(self, batch, model_outputs):
        # MONITORING LOSS
        targets_per_timestep = self._reshape_latents_per_timestep(
            batch["target_latents"]
        )
        predictions_per_timestep = self._reshape_latents_per_timestep(
            model_outputs["predicted_latents"]
        )

        monitor_value_raw = self.monitor_loss_fn(
            predictions_per_timestep.detach(), targets_per_timestep.detach()
        ).detach()  # [B, T, D*H*W]

        monitor_per_timestep = monitor_value_raw.mean(dim=[0, 2])

        # Also monitor per sample
        l1_per_sample = monitor_value_raw.mean(dim=[1, 2])  # [B]

        # Build the result dictionary
        result = {
            self.monitor_loss_name: monitor_per_timestep.mean(),  # Overall average
            f"{self.monitor_loss_name}_per_sample": l1_per_sample,  # Per sample average
        }

        # Add one entry per timestep
        for t in range(monitor_per_timestep.shape[0]):
            result[f"{self.monitor_loss_name}_t{t}"] = monitor_per_timestep[t]

        return result


class DownstreamLoss:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn  # This is typically l1 loss

    def _prepare_inputs_and_predictions_for_downstream_task(self, targets, predictions):
        targets = targets.squeeze(-1)  # [B, T, H, W]
        predictions = predictions.squeeze(2)  # [B, T, H, W]
        height, width = predictions.shape[-2:]
        flat_predictions = predictions.reshape(
            predictions.shape[0], predictions.shape[1], -1
        )
        max_idx_pred = flat_predictions.argmax(dim=2)
        pred_y = max_idx_pred // width
        pred_x = max_idx_pred % width
        predicted_coords = torch.stack((pred_x, pred_y), dim=2)  # [B, T, 2]
        flat_targets = targets.reshape(targets.shape[0], targets.shape[1], -1)
        max_idx_target = flat_targets.argmax(dim=2)
        target_y = max_idx_target // width
        target_x = max_idx_target % width
        target_coords = torch.stack((target_x, target_y), dim=2)  # [B, T, 2]
        return target_coords, predicted_coords

    def _get_future_targets(self, batch, num_context_latents, num_target_latents):
        targets = {}
        for shape in ("square", "circle"):
            key = f"target_{shape}_heatmap_pixel"
            target = batch[key]  # [B, num_frames_video, img_size, img_size, 1]
            target = target[
                :, num_context_latents : num_context_latents + num_target_latents
            ]  # [B, num_target_latents, img_size, img_size, 1]
            targets[key] = target  # [B, num_target_latents, img_size, img_size, 1]
        return targets

    def compute_loss(
        self, batch, model_outputs, num_context_latents, num_target_latents
    ):
        targets = self._get_future_targets(
            batch, num_context_latents, num_target_latents
        )
        losses = {}
        for stem in ("oracle", "predicted"):
            stem_losses = []
            for shape in ("square", "circle"):
                target = targets[
                    f"target_{shape}_heatmap_pixel"
                ]  # [B, num_target_latents, img_size, img_size, 1]
                predictions = model_outputs[
                    f"{stem}_{shape}_heatmap_pixel"
                ]  # [B, num_target_latents, 1, img_size, img_size]
                target_coords, predicted_coords = (
                    self._prepare_inputs_and_predictions_for_downstream_task(
                        target, predictions
                    )
                )
                loss = self.loss_fn(predicted_coords.float(), target_coords.float())
                losses[f"downstream_loss_{stem}_{shape}_heatmap_pixel"] = loss
                stem_losses.append(loss)
            stem_total = torch.stack(stem_losses).mean()
            if stem == "oracle":
                losses["downstream_heatmap_pixel_l1_oracle"] = stem_total
            else:
                losses["downstream_heatmap_pixel_l1_predicted"] = stem_total
        return losses


class GramAnchoringLoss(BaseLoss):
    """Gram anchoring loss for matching patch similarity structure."""

    def __init__(
        self,
        weight_loss: float = 1.0,
        loss_type: str = "l2",
        temporal: bool = False,
        time_weighting: str = "none",
        tau_min: float = 0.0,
        tau_max: float = 1.0,
    ):
        """
        Args:
            weight_loss: Weight multiplier for the loss
            loss_type: "l1" or "l2" for comparing Gram matrices
            temporal: If False, compute Gram per frame [B, T, N, N]
                      If True, compute Gram across all spacetime tokens [B, T*N, T*N]
            tau_min: Minimum timestep (inclusive) for computing loss
            tau_max: Maximum timestep (inclusive) for computing loss
        """
        self.weight_loss = weight_loss
        if loss_type not in ("l1", "l2"):
            raise ValueError(f"loss_type must be 'l1' or 'l2', got '{loss_type}'")
        self.loss_type = loss_type
        self.temporal = temporal
        if time_weighting not in ("none", "linear", "squared", "inverse_linear"):
            raise ValueError(
                f"time_weighting must be one of 'none', 'linear', 'squared', 'inverse_linear', "
                f"got '{time_weighting}'"
            )
        self.time_weighting = time_weighting
        if not (0.0 <= tau_min <= 1.0):
            raise ValueError(f"tau_min must be in [0, 1], got '{tau_min}'")
        if not (0.0 <= tau_max <= 1.0):
            raise ValueError(f"tau_max must be in [0, 1], got '{tau_max}'")
        if tau_min > tau_max:
            raise ValueError(
                f"tau_min must be <= tau_max, got '{tau_min}' > '{tau_max}'"
            )
        self.tau_min = tau_min
        self.tau_max = tau_max

    def _compute_gram_matrix(self, x, normalize_d):
        """
        x: tensor [B, M, D] where M is number of tokens (either N or T*N)
        return -> gram: tensor [B, M, M] (similarity matrix)
        """
        gram_matrix = torch.matmul(x, x.transpose(-1, -2))
        if normalize_d:
            D = x.shape[-1]
            gram_matrix = gram_matrix / D
        return gram_matrix

    def _compute_time_weights(self, t):
        """
        Compute time-based weights from timestep tensor t.
        Args:
            t: tensor [B] with timestep values
        Returns:
            weights: tensor [B] with computed weights
        """
        if self.time_weighting == "linear":
            return t
        elif self.time_weighting == "squared":
            return t**2
        elif self.time_weighting == "inverse_linear":
            return 1.0 - t
        else:
            raise ValueError(f"Unknown time_weighting: {self.time_weighting}")

    def _compute_time_mask(self, t):
        mask = (t >= self.tau_min) & (t <= self.tau_max)
        return mask.float()

    def compute_loss(self, batch, model_outputs, normalize_d, t=None):
        target = batch["target_latents"]
        pred = model_outputs["predicted_latents"]

        if self.temporal:
            # Flatten space AND time: [B, D, T, H, W] -> [B, T*H*W, D]
            target_tokens = rearrange(target, "b d t h w -> b (t h w) d")
            pred_tokens = rearrange(pred, "b d t h w -> b (t h w) d")
        else:
            # Flatten space only: [B, D, T, H, W] -> [B, T, H*W, D]
            target_tokens = rearrange(target, "b d t h w -> b t (h w) d")
            pred_tokens = rearrange(pred, "b d t h w -> b t (h w) d")

        B = target.shape[0]

        gram_target = self._compute_gram_matrix(target_tokens, normalize_d)
        gram_pred = self._compute_gram_matrix(pred_tokens, normalize_d)

        # Upper triangle mask
        M = gram_target.shape[-1]
        triu_mask = torch.triu(
            torch.ones(M, M, device=gram_target.device, dtype=torch.bool)
        )

        gram_diff = gram_pred - gram_target
        gram_diff_triu = gram_diff[..., triu_mask]

        # Compute loss per batch element: [B, ...] -> [B]
        if self.loss_type == "l2":
            loss_per_batch = (gram_diff_triu**2).flatten(start_dim=1).mean(dim=1)
        else:  # l1
            loss_per_batch = gram_diff_triu.abs().flatten(start_dim=1).mean(dim=1)
        assert loss_per_batch.shape == (B,)

        if self.time_weighting == "none":
            weighted_loss_per_batch = loss_per_batch  # [B]
        else:
            time_weights = self._compute_time_weights(t)  # [B]
            weighted_loss_per_batch = loss_per_batch * time_weights  # [B]

        if t is not None:
            time_mask = self._compute_time_mask(
                t
            )  # tells which batch elements to include
            if time_mask.any().item():
                loss = (weighted_loss_per_batch * time_mask).sum() / time_mask.sum()
            else:
                loss = weighted_loss_per_batch.sum() * 0.0
        else:
            loss = weighted_loss_per_batch.mean()

        return {"loss_gram": loss * self.weight_loss}


class TemporalConsistencyLoss(BaseLoss):
    """Temporal consistency loss comparing frame-to-frame differences."""

    def __init__(self, weight_loss: float = 1.0, loss_type: str = "l1"):
        """
        Args:
            weight_loss: Weight multiplier for the loss
            loss_type: "l1" or "l2" for comparing temporal differences
        """
        self.weight_loss = weight_loss
        if loss_type not in ("l1", "l2"):
            raise ValueError(f"loss_type must be 'l1' or 'l2', got '{loss_type}'")
        self.loss_type = loss_type

    def _compute_temporal_diff(self, x):
        """
        Compute frame-to-frame differences.

        Args:
            x: tensor [B, D, T, H, W]

        Returns:
            temporal_diff: tensor [B, D, T-1, H, W]
        """
        return x[:, :, 1:] - x[:, :, :-1]

    def compute_loss(self, batch, model_outputs):
        """
        Compute temporal consistency loss.

        Args:
            batch: dict containing target_latents
            model_outputs: dict containing predicted_latents

        Returns:
            dict with loss_consistency
        """
        target = batch["target_latents"]
        pred = model_outputs["predicted_latents"]

        # Compute temporal differences [B, D, T-1, H, W]
        target_diff = self._compute_temporal_diff(target)
        pred_diff = self._compute_temporal_diff(pred)

        # Flatten for loss computation
        target_diff_flat = rearrange(target_diff, "b d t h w -> b (t d h w)")
        pred_diff_flat = rearrange(pred_diff, "b d t h w -> b (t d h w)")

        diff = pred_diff_flat - target_diff_flat

        if self.loss_type == "l2":
            loss = torch.mean(diff**2)
        else:  # l1
            loss = torch.mean(torch.abs(diff))

        return {"loss_consistency": loss * self.weight_loss}


class SIGRegFlowMatching(BaseLoss):
    def __init__(
        self, knots=34, max_t=3.0, n_projections=256, frame_level: bool = True
    ):
        # initialize the times:
        self.max_t = max_t
        self._t = torch.linspace(-max_t, max_t, knots)  # [K]
        self._weights = torch.exp(-0.5 * torch.square(self._t))  # [K]
        self.M = n_projections
        self.frame_level = frame_level

    def _sample_random_directions(self, D, device, dtype=None):
        A = torch.randn(size=(D, self.M), device=device, dtype=dtype)
        A /= A.norm(dim=0)
        return A  # [D, M]

    def compute_loss(self, batch, model_outputs):
        target = batch["target_latents"]
        pred = model_outputs["predicted_latents"]

        # target, pred: [B, D, T, H, W]
        B, D, T, H, W = target.shape
        device = target.device
        dtype = target.dtype

        if self.frame_level:
            # per-frame sets of tokens: [(B*T), N, D], with N = H*W
            target = rearrange(target, "b d t h w -> (b t) (h w) d")
            pred = rearrange(pred, "b d t h w -> (b t) (h w) d")
            N = H * W
        else:
            # per-video sets of tokens: [B, N, D], with N = T*H*W
            target = rearrange(target, "b d t h w -> b (t h w) d")
            pred = rearrange(pred, "b d t h w -> b (t h w) d")
            N = T * H * W

        # move buffers to device/dtype (minimal + safe)
        t = self._t.to(device=device, dtype=dtype)  # [K]
        w = self._weights.to(device=device, dtype=dtype)  # [K]

        A = self._sample_random_directions(D, device, dtype)  # [D, M]

        # projections: [(B*T), N, M]
        target_proj = target @ A
        pred_proj = pred @ A

        # evaluate on frequency grid: [(B*T), N, M, K]
        target_phase = target_proj.unsqueeze(-1) * t
        pred_phase = pred_proj.unsqueeze(-1) * t

        # ECF components: average over tokens N (dim=1) => [(B*T), M, K]
        target_cos = torch.cos(target_phase).mean(dim=1)
        target_sin = torch.sin(target_phase).mean(dim=1)
        pred_cos = torch.cos(pred_phase).mean(dim=1)
        pred_sin = torch.sin(pred_phase).mean(dim=1)

        # squared CF distance per (frame, projection, freq): [(B*T), M, K]
        err = (pred_cos - target_cos).square() + (pred_sin - target_sin).square()

        # apply Gaussian window over frequencies (broadcast on last dim)
        err = err * w

        # integrate over frequencies K -> [(B*T), M]
        # (trapz handles the dt weights correctly for your grid)
        stat = torch.trapz(err, t, dim=-1)

        # average over projections M and frames (B*T) -> scalar
        loss = stat.mean()

        # Optional: if you want the "test statistic" scaling like in SIGReg papers:
        # loss = loss * N   # (careful: can be huge for tokens)

        return {"loss_sigreg": loss}
