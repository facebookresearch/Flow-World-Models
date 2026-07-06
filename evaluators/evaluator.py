import contextlib
import io
import json
import logging
import os
from collections import defaultdict
from time import time
import pandas as pd

import torch
import torch.distributed as dist
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .ranked_cocoeval import RankedCOCOeval


def _recursive_to_cpu(data):
    """Recursively move tensors in nested dicts/lists to CPU (detached)."""
    if isinstance(data, torch.Tensor):
        return data.detach().to("cpu")
    if isinstance(data, dict):
        return {k: _recursive_to_cpu(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_recursive_to_cpu(v) for v in data]
    return data


# Used only in the DeterministicPredictorEvaluatorWaymo evaluator
def evaluate_coco_detections(
    predictions_path, gt_annotations_path=None, logger=None, coco_gt=None
):
    """
    Evaluate COCO detections using pycocotools.

    Args:
        predictions_path: Path to predictions COCO format JSON.
        gt_annotations_path: Path to ground-truth COCO annotations (if coco_gt not provided).
        logger: Optional logger for output.
        coco_gt: Pre-loaded COCO ground-truth instance to reuse across evaluations.
    """
    if predictions_path is None:
        raise ValueError("`predictions_path` must be provided for COCO evaluation.")
    if coco_gt is None:
        if gt_annotations_path is None:
            raise ValueError(
                "Either `coco_gt` or `gt_annotations_path` must be provided."
            )
        coco_gt = COCO(gt_annotations_path)

    coco_dt = coco_gt.loadRes(predictions_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "AP": coco_eval.stats[0] * 100,
        "AP_50": coco_eval.stats[1] * 100,
        "AP_75": coco_eval.stats[2] * 100,
        "AP_small": coco_eval.stats[3] * 100,
        "AP_medium": coco_eval.stats[4] * 100,
        "AP_large": coco_eval.stats[5] * 100,
        "AR_1": coco_eval.stats[6] * 100,
        "AR_10": coco_eval.stats[7] * 100,
        "AR_100": coco_eval.stats[8] * 100,
        "AR_small": coco_eval.stats[9] * 100,
        "AR_medium": coco_eval.stats[10] * 100,
        "AR_large": coco_eval.stats[11] * 100,
    }

    if logger:
        logger.info("=" * 60)
        logger.info("COCO Evaluation Results:")
        logger.info("=" * 60)
        logger.info(f"  AP @ IoU=0.50:0.95: {metrics['AP']:.4f}")
        logger.info(f"  AP small: {metrics['AP_small']:.4f}")
        logger.info(f"  AP medium: {metrics['AP_medium']:.4f}")
        logger.info(f"  AP large: {metrics['AP_large']:.4f}")
        logger.info("=" * 60)

    return metrics


class BaseEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg

    def _log_final_metrics(
        self, num_batches, eval_start_time, coco_metrics, oracle_metrics
    ):
        """Computes averages, compiles the summary dict, and logs to WandB/Console."""
        # 1. Average Latent Loss
        avg_l1 = (
            self.epoch_running_losses.get(
                "l1_latents", torch.tensor(0.0, device=self.device)
            )
            / num_batches
        )
        dist.all_reduce(avg_l1)
        avg_l1 /= dist.get_world_size()

        # 2. Average Timestep Losses
        avg_per_timestep = {}
        for key, val in self.epoch_running_losses.items():
            if key.startswith("l1_latents_t"):
                t_loss = val / num_batches
                dist.all_reduce(t_loss)
                avg_per_timestep[key] = (t_loss / dist.get_world_size()).item()

        # 3. Evaluator Metrics (Synthetic)
        eval_metrics = self.evaluator.evaluate() if self.evaluator is not None else {}

        # 4. Construct Summary Dictionary
        summary = {
            "eval/l1_latents_loss": avg_l1.item(),
            "eval/duration_sec": time() - eval_start_time,
            **{f"eval/{k}": v for k, v in avg_per_timestep.items()},
        }

        # Add Evaluator metrics
        if eval_metrics:
            summary.update(
                {
                    "eval/prediction_precision_error": eval_metrics.get(
                        "prediction_precision_error"
                    ),
                    "eval/ground_truth_recall_error": eval_metrics.get(
                        "ground_truth_recall_error"
                    ),
                    "eval/f1_error": eval_metrics.get("f1_error"),
                }
            )

        # Add COCO metrics
        if coco_metrics:
            summary.update(
                {f"eval/coco/{k.lower()}": v for k, v in coco_metrics.items()}
            )
        if oracle_metrics:
            summary.update(
                {f"eval/coco_oracle/{k.lower()}": v for k, v in oracle_metrics.items()}
            )

        # 5. Log to WandB
        if self.global_rank == 0:
            if self.wandb_run:
                self.wandb_run.log(summary, step=self.train_state.gradient_steps)

            # 6. Console Log construction
            log_parts = [
                f"[EVAL] epoch {self.train_state.epoch}",
                f"L1 loss={summary['eval/l1_latents_loss']:.4f}",
            ]
            if "eval/f1_error" in summary:
                log_parts.append(f"F1 err={summary['eval/f1_error']:.4f}")
            if coco_metrics:
                log_parts.append(f"AP={coco_metrics.get('AP', 0):.4f}")
            if oracle_metrics:
                log_parts.append(f"Oracle AP={oracle_metrics.get('AP', 0):.4f}")
            log_parts.append(f"duration={summary['eval/duration_sec']:.2f}s")

            self.logger.info(" | ".join(log_parts))


class DeterministicPredictorEvaluatorSyntheticBouncingShapes(BaseEvaluator):
    """Minimal evaluator that stores GT and predictions as (x,y) positions."""

    def __init__(self, cfg=None, name: str = "deterministic_predictor_evaluator"):
        super().__init__(cfg)
        self.name = name
        # Ground truths: { ic_id: { traj_id: {"square": Tensor[T,2], "circle": Tensor[T,2]} } }
        self._ground_truths = defaultdict(dict)
        # Predictions (unique per initial condition): { ic_id: {"square": Tensor[T,2], "circle": Tensor[T,2]} }
        self._predictions = {}

    @staticmethod
    def _heatmap_to_xy_coords(heatmap: torch.Tensor) -> torch.Tensor:
        height, width = heatmap.shape[-2:]
        flat = heatmap.reshape(
            *heatmap.shape[:-2], -1
        )  # [B, num_frames, img_size * img_size]
        max_idx = flat.argmax(dim=-1)  # [B, num_frames]
        y = max_idx // width
        x = max_idx % width
        return torch.stack((x, y), dim=-1)

    @staticmethod
    def _to_cpu(data):
        if isinstance(data, torch.Tensor):
            return data.detach().to("cpu")
        if isinstance(data, dict):
            return {
                k: _recursive_to_cpu(v) for k, v in data.items()
            }
        if isinstance(data, list):
            return [_recursive_to_cpu(v) for v in data]
        return data

    def extract_xy_positions_from_gt_one_hot_targets(
        self, square_heatmap: torch.Tensor, circle_heatmap: torch.Tensor
    ):
        square_heatmap = square_heatmap.squeeze(
            -1
        )  # [B, num_ctxt_frames + num_target_frames, img_size, img_size]
        circle_heatmap = circle_heatmap.squeeze(
            -1
        )  # [B, num_ctxt_frames + num_target_frames, img_size, img_size]
        square_coords = self._heatmap_to_xy_coords(square_heatmap)  # [B, num_frames, 2]
        circle_coords = self._heatmap_to_xy_coords(circle_heatmap)  # [B, num_frames, 2]
        return square_coords, circle_coords

    def extract_xy_positions_from_pred_logits(
        self, square_heatmap: torch.Tensor, circle_heatmap: torch.Tensor
    ):
        square_heatmap = square_heatmap.squeeze(2)
        circle_heatmap = circle_heatmap.squeeze(2)
        square_coords = self._heatmap_to_xy_coords(square_heatmap)
        circle_coords = self._heatmap_to_xy_coords(circle_heatmap)
        return square_coords, circle_coords

    def add_gt_and_predictions(self, batch, model_outputs):
        # Store the GT
        gt_square_heatmap_pixel = batch[
            "target_square_heatmap_pixel"
        ]  # [B, num_ctxt_frames + num_target_frames, img_size, img_size, 1]
        gt_circle_heatmap_pixel = batch[
            "target_circle_heatmap_pixel"
        ]  # [B, num_ctxt_frames + num_target_frames, img_size, img_size, 1]
        # store predictions
        predicted_square_heatmap_pixel = model_outputs[
            "predicted_square_heatmap_pixel"
        ]  # [B, num_ctxt_frames, 1, img_size, img_size]
        predicted_circle_heatmap_pixel = model_outputs[
            "predicted_circle_heatmap_pixel"
        ]  # [B, num_ctxt_frames, 1, img_size, img_size]

        # For now these predictions are one hot targets (for ground truths, and logits for the predictions). But we need to convert them in (x,y) positions for both
        gt_square_xy_positions, gt_circle_xy_positions = (
            self.extract_xy_positions_from_gt_one_hot_targets(
                gt_square_heatmap_pixel, gt_circle_heatmap_pixel
            )
        )  # [B, num_frames, 2] each
        pred_square_xy_positions, pred_circle_xy_positions = (
            self.extract_xy_positions_from_pred_logits(
                predicted_square_heatmap_pixel, predicted_circle_heatmap_pixel
            )
        )  # [B, num_frames, 2] each

        num_target_frames = pred_square_xy_positions.shape[1]
        gt_square_xy_positions = gt_square_xy_positions[:, -num_target_frames:]
        gt_circle_xy_positions = gt_circle_xy_positions[:, -num_target_frames:]

        # --- 2) Metadata on CPU ---
        meta = self._to_cpu(batch["metadata"])
        ic_ids = meta["initial_conditions_id"]  # list[str] of length B
        traj_ids = meta["trajectory_id"]  # list[str] of length B

        # --- 3) Store per-sample ---
        B = gt_square_xy_positions.shape[0]
        for i in range(B):
            ic_id = ic_ids[i]
            traj_id = traj_ids[i]

            # (A) Ground truths: double index (ic_id, traj_id)
            if traj_id in self._ground_truths[ic_id]:
                # reverse
                raise ValueError(
                    f"Initial condition ID '{ic_id}' with trajectory ID '{traj_id}' was already stored in ground truths."
                )
            self._ground_truths[ic_id][traj_id] = {
                "square_xy_positions": gt_square_xy_positions[i]
                .detach()
                .cpu(),  # [T, 2]
                "circle_xy_positions": gt_circle_xy_positions[i]
                .detach()
                .cpu(),  # [T, 2]
            }

            # (B) Predictions: one per ic_id; if already present, skip
            if ic_id not in self._predictions:
                self._predictions[ic_id] = {
                    "square_xy_positions": pred_square_xy_positions[i]
                    .detach()
                    .cpu(),  # [T, 2]
                    "circle_xy_positions": pred_circle_xy_positions[i]
                    .detach()
                    .cpu(),  # [T, 2]
                }
            # else: prediction for this ic_id already stored; skip

    def accumulate(self, batch, model_outputs, downstream_outputs=None):
        """Trainer-facing alias for add_gt_and_predictions.

        Heatmap predictions are appended to ``model_outputs`` by the downstream
        pass, so ``downstream_outputs`` is unused for this evaluator.
        """
        self.add_gt_and_predictions(batch, model_outputs)

    # Optional helpers to fetch what was stored (minimal)
    def get_ground_truths(self):
        return self._ground_truths

    def get_predictions(self):
        return self._predictions

    def reset_evaluator(self):
        # Clear stored data
        self._ground_truths = defaultdict(dict)
        self._predictions = {}

    def evaluate(self, *args, **kwargs):
        """Basic evaluation scaffold that checks predictions/GT counts and iterates per context."""
        self.synchronize_across_processes()  # This will gather all data from all processes
        if dist.get_rank() == 0:
            metrics = self.compute_metrics()  # Compute metrics
        else:
            metrics = None
        # Prepare evaluator for the next evaluation call.
        self.reset_evaluator()
        return metrics

    def compute_precision(self):
        errors = {}
        for ic_id, prediction in self._predictions.items():
            ground_truths = self._ground_truths[ic_id]
            best_error = None
            for gt in ground_truths.values():
                sq_err = (
                    (prediction["square_xy_positions"] - gt["square_xy_positions"])
                    .abs()
                    .float()
                    .mean()
                )
                cir_err = (
                    (prediction["circle_xy_positions"] - gt["circle_xy_positions"])
                    .abs()
                    .float()
                    .mean()
                )
                error = 0.5 * (sq_err + cir_err)
                best_error = (
                    error if best_error is None or error < best_error else best_error
                )
            errors[ic_id] = best_error.item()
        return errors

    def compute_recall(self):
        errors = {}
        for ic_id, trajectories in self._ground_truths.items():
            prediction = self._predictions[ic_id]
            ic_errors = {}
            for traj_id, gt in trajectories.items():
                sq_err = (
                    (prediction["square_xy_positions"] - gt["square_xy_positions"])
                    .abs()
                    .float()
                    .mean()
                )
                cir_err = (
                    (prediction["circle_xy_positions"] - gt["circle_xy_positions"])
                    .abs()
                    .float()
                    .mean()
                )
                error = 0.5 * (sq_err + cir_err)
                ic_errors[traj_id] = error.item()
            errors[ic_id] = ic_errors
        return errors

    def compute_metrics(self):
        """Compute per-initial-condition L1 metrics for each stored trajectory."""
        prediction_precision_errors = self.compute_precision()
        ground_truth_recall_errors = self.compute_recall()
        prediction_precision_error = (
            torch.tensor(
                list(prediction_precision_errors.values()), dtype=torch.float32
            )
            .mean()
            .item()
        )
        ground_truth_recall_error = (
            torch.tensor(
                [
                    err
                    for ic_errors in ground_truth_recall_errors.values()
                    for err in ic_errors.values()
                ],
                dtype=torch.float32,
            )
            .mean()
            .item()
        )
        denom = prediction_precision_error + ground_truth_recall_error
        f1_error = 2 * prediction_precision_error * ground_truth_recall_error / denom
        return {
            "prediction_precision_error": prediction_precision_error,
            "ground_truth_recall_error": ground_truth_recall_error,
            "f1_error": f1_error,
            "prediction_precision_errors": prediction_precision_errors,
            "ground_truth_recall_errors": ground_truth_recall_errors,
        }

    def synchronize_across_processes(self):
        if not dist.is_available() or not dist.is_initialized():
            return
        state = {
            "ground_truths": dict(self._ground_truths),
            "predictions": dict(self._predictions),
        }
        # Count how many initial conditions in the ground truths, this the number of keys
        keys = list(state["ground_truths"].keys())
        num_initial_conditions = len(keys)

        # Now count the number of total trajectories in the ground truths
        num_trajectories = sum(len(trajs) for trajs in state["ground_truths"].values())

        # Now print the number of keys in the predictions
        num_prediction_keys = len(state["predictions"].keys())

        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, state)
        merged_gt = defaultdict(dict)
        merged_pred = {}
        for item in gathered:
            for ic_id, trajectories in item["ground_truths"].items():
                for traj_id, data in trajectories.items():
                    if traj_id in merged_gt[ic_id]:
                        raise ValueError(
                            f"Duplicate trajectory id '{traj_id}' for context '{ic_id}' across ranks."
                        )
                    merged_gt[ic_id][traj_id] = data
            for ic_id, data in item["predictions"].items():
                merged_pred[ic_id] = data

        self._ground_truths = merged_gt
        self._predictions = merged_pred


class DeterministicPredictorEvaluatorWaymo(BaseEvaluator):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.det_cfg = cfg.get("evaluation_detection", None)
        self.list_coco_predictions = []
        self.list_coco_predictions_oracle = []

    def accumulate(self, batch, model_outputs, downstream_outputs):
        coco_predictions = downstream_outputs["coco_predictions"]
        oracle_predictions = downstream_outputs["coco_predictions_oracle"]

        self.list_coco_predictions.extend(coco_predictions)
        self.list_coco_predictions_oracle.extend(oracle_predictions)

    def _process_and_evaluate_coco(
        self, predictions_list, filename, dump_dir, description="COCO"
    ):
        """
        Handles the distributed gather, merge, save, and evaluation for COCO predictions.
        Returns the metrics dictionary or None.
        """

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # 1. Gather from all ranks
        if dist.is_initialized():
            gathered = [None for _ in range(world_size)]
            dist.gather_object(predictions_list, gathered if rank == 0 else None, dst=0)
        else:
            gathered = [predictions_list]

        eval_results = None
        if rank == 0:
            # 2. Merge
            merged_preds = [p for sub in gathered for p in sub]
            out_path = os.path.join(dump_dir, filename)

            # 3. Save to JSON
            with open(out_path, "w") as f:
                json.dump(merged_preds, f)
                f.flush()

            # 4. Run Evaluation
            eval_results = evaluate_coco_detections(
                predictions_path=out_path,
                gt_annotations_path=self.det_cfg.path_annotations,
            )

        if dist.is_initialized():
            dist.barrier()
        return eval_results

    def evaluate(self, dump_epoch_dir):
        """
        Aggregates COCO predictions collected via `accumulate`, saves them to disk, and runs
        Detectron style evaluation. Returns a tuple (standard_metrics, oracle_metrics).
        """
        coco_metrics = self._process_and_evaluate_coco(
            self.list_coco_predictions,
            "coco_detections.json",
            dump_epoch_dir,
            description="COCO",
        )

        oracle_metrics = self._process_and_evaluate_coco(
            self.list_coco_predictions_oracle,
            "coco_detections_oracle.json",
            dump_epoch_dir,
            description="Oracle COCO",
        )

        # Reset for next evaluation cycle
        self.list_coco_predictions.clear()
        self.list_coco_predictions_oracle.clear()

        evaluation_metrics = {
            "coco_metrics": coco_metrics,
            "oracle_metrics": oracle_metrics,
        }

        return evaluation_metrics


# Now create evaluator for flow matching
class FlowMatchingEvaluatorSyntheticBouncingShapes(BaseEvaluator):
    """Minimal evaluator that stores GT and predictions as (x,y) positions."""

    def __init__(self, cfg=None, name: str = "flow_matching_evaluator"):
        super().__init__(cfg)
        self.name = name
        # Ground truths: { ic_id: { traj_id: {"square": Tensor[T,2], "circle": Tensor[T,2]} } }
        self._ground_truths = defaultdict(dict)
        # Predictions (triple indexing): { ic_id: { traj_id: { pred_idx: {"square": Tensor[T,2], "circle": Tensor[T,2]} } } }
        self._predictions = defaultdict(lambda: defaultdict(dict))

    @staticmethod
    def _heatmap_to_xy_coords(heatmap: torch.Tensor) -> torch.Tensor:
        height, width = heatmap.shape[-2:]
        flat = heatmap.reshape(
            *heatmap.shape[:-2], -1
        )  # [B, num_frames, img_size * img_size]
        max_idx = flat.argmax(dim=-1)  # [B, num_frames]
        y = max_idx // width
        x = max_idx % width
        return torch.stack((x, y), dim=-1)

    @staticmethod
    def _to_cpu(data):
        if isinstance(data, torch.Tensor):
            return data.detach().to("cpu")
        if isinstance(data, dict):
            return {
                k: _recursive_to_cpu(v) for k, v in data.items()
            }
        if isinstance(data, list):
            return [_recursive_to_cpu(v) for v in data]
        return data

    def extract_xy_positions_from_gt_one_hot_targets(
        self, square_heatmap: torch.Tensor, circle_heatmap: torch.Tensor
    ):
        square_heatmap = square_heatmap.squeeze(
            -1
        )  # [B, num_ctxt_frames + num_target_frames, img_size, img_size]
        circle_heatmap = circle_heatmap.squeeze(
            -1
        )  # [B, num_ctxt_frames + num_target_frames, img_size, img_size]
        square_coords = self._heatmap_to_xy_coords(square_heatmap)  # [B, num_frames, 2]
        circle_coords = self._heatmap_to_xy_coords(circle_heatmap)  # [B, num_frames, 2]
        return square_coords, circle_coords

    def extract_xy_positions_from_pred_logits(
        self, square_heatmap: torch.Tensor, circle_heatmap: torch.Tensor
    ):
        # heatmaps have shape [B, num_tgt_frames, img_size, img_size]
        square_coords = self._heatmap_to_xy_coords(square_heatmap)
        circle_coords = self._heatmap_to_xy_coords(circle_heatmap)
        return square_coords, circle_coords

    def add_gt_and_predictions(
        self, batch, list_of_temporary_predictions_to_be_saved_in_evaluator
    ):
        # Store the GT
        gt_square_heatmap_pixel = batch[
            "target_square_heatmap_pixel"
        ]  # [B, num_ctxt_frames + num_target_frames, img_size, img_size, 1]
        gt_circle_heatmap_pixel = batch[
            "target_circle_heatmap_pixel"
        ]  # [B, num_ctxt_frames + num_target_frames, img_size, img_size, 1]

        # For now these predictions are one hot targets (for ground truths, and logits for the predictions). But we need to convert them in (x,y) positions for both
        gt_square_xy_positions, gt_circle_xy_positions = (
            self.extract_xy_positions_from_gt_one_hot_targets(
                gt_square_heatmap_pixel, gt_circle_heatmap_pixel
            )
        )  # [B, num_frames, 2] each

        num_target_frames = 16  # temporary fix
        gt_square_xy_positions = gt_square_xy_positions[:, -num_target_frames:]
        gt_circle_xy_positions = gt_circle_xy_positions[:, -num_target_frames:]

        meta = batch["metadata"]
        ic_ids = meta["initial_conditions_id"]  # list[str] of length B
        traj_ids = meta["trajectory_id"]  # list[str] of length B
        assert len(ic_ids) == len(traj_ids)

        # --- Store per-sample ground truths ---
        B = gt_square_xy_positions.shape[0]
        for i in range(B):
            ic_id = ic_ids[i]
            traj_id = traj_ids[i]

            if traj_id in self._ground_truths[ic_id]:
                print(
                    f"Initial condition ID '{ic_id}' with trajectory ID '{traj_id}' was already stored in ground truths. Skipping."
                )
            else:
                self._ground_truths[ic_id][traj_id] = {
                    "square_xy_positions": gt_square_xy_positions[i]
                    .detach()
                    .cpu(),  # [T, 2]
                    "circle_xy_positions": gt_circle_xy_positions[i]
                    .detach()
                    .cpu(),  # [T, 2]
                }

        # --- Store predictions with triple indexing ---
        for i, model_outputs in enumerate(
            list_of_temporary_predictions_to_be_saved_in_evaluator
        ):
            pred_idx = model_outputs["prediction_idx"]  # int
            predicted_square_heatmap_pixel = model_outputs[
                "predicted_square_heatmap_pixel"
            ]  # [B, num_tgt_frames, img_size, img_size]
            predicted_circle_heatmap_pixel = model_outputs[
                "predicted_circle_heatmap_pixel"
            ]  # [B, num_tgt_frames, img_size, img_size]

            pred_square_xy_positions, pred_circle_xy_positions = (
                self.extract_xy_positions_from_pred_logits(
                    predicted_square_heatmap_pixel, predicted_circle_heatmap_pixel
                )
            )  # [B, num_target_frames, 2] each

            for b in range(B):
                ic_id = model_outputs["initial_conditions_id"][b]  # str
                traj_id = model_outputs["trajectory_id"][b]  # str

                # check that empty otherwise raise Error
                if pred_idx in self._predictions[ic_id][traj_id]:
                    raise ValueError(
                        f"Initial condition ID '{ic_id}' with trajectory ID '{traj_id}' and prediction index '{pred_idx}' was already stored in predictions."
                    )

                # Triple indexing: [ic_id][traj_id][pred_idx]
                self._predictions[ic_id][traj_id][pred_idx] = {
                    "square_xy_positions": pred_square_xy_positions[b].detach().cpu(),
                    "circle_xy_positions": pred_circle_xy_positions[b].detach().cpu(),
                }

    def get_ground_truths(self):
        return self._ground_truths

    def get_predictions(self):
        return self._predictions

    def check_gt_and_prediction_counts(self):
        """Check that for each initial condition, the number of predictions and ground truths are consistent."""
        # Compute number of initial conditions for gt and predictions, print them and check they match
        num_ic_gt = len(self._ground_truths)
        num_ic_pred = len(self._predictions)
        assert num_ic_gt == num_ic_pred, (
            f"Mismatch in number of initial conditions: GT={num_ic_gt}, Pred={num_ic_pred}"
        )

        # Compute total number of ground truths
        total_gt = sum(len(trajs) for trajs in self._ground_truths.values())
        total_pred = sum(
            sum(len(preds) for preds in trajs.values())
            for trajs in self._predictions.values()
        )

        # Check that total number of predictions is twice the number of total GT
        expected_pred = 2 * total_gt
        assert total_pred == expected_pred, (
            f"Expected {expected_pred} predictions (2x GT), but found {total_pred}"
        )

    def reset_evaluator(self):
        self._ground_truths = defaultdict(dict)
        self._predictions = defaultdict(lambda: defaultdict(dict))

    def evaluate(self, *args, **kwargs):
        """Basic evaluation scaffold that checks predictions/GT counts and iterates per context."""
        self.synchronize_across_processes()  # This will gather all data from all processes
        if dist.get_rank() == 0:
            self.check_gt_and_prediction_counts()
            metrics = self.compute_metrics()  # Compute metrics
        else:
            metrics = None
        # Prepare evaluator for the next evaluation call.
        self.reset_evaluator()
        return metrics

    def compute_precision(self):
        """For each prediction, find the closest GT trajectory with the same context."""
        errors = {}
        for ic_id in self._predictions:
            ground_truths = self._ground_truths[ic_id]
            for traj_id in self._predictions[ic_id]:
                for pred_idx, prediction in self._predictions[ic_id][traj_id].items():
                    best_error = None
                    for gt in ground_truths.values():
                        sq_err = (
                            (
                                prediction["square_xy_positions"]
                                - gt["square_xy_positions"]
                            )
                            .abs()
                            .float()
                            .mean()
                        )
                        cir_err = (
                            (
                                prediction["circle_xy_positions"]
                                - gt["circle_xy_positions"]
                            )
                            .abs()
                            .float()
                            .mean()
                        )
                        error = 0.5 * (sq_err + cir_err)
                        best_error = (
                            error
                            if best_error is None or error < best_error
                            else best_error
                        )
                    errors[(ic_id, traj_id, pred_idx)] = best_error.item()
        return errors

    def compute_recall(self):
        """For each GT trajectory, find the closest prediction with the same context (any traj_id, any pred_idx)."""
        errors = {}
        for ic_id, trajectories in self._ground_truths.items():
            ic_errors = {}
            for traj_id, gt in trajectories.items():
                best_error = None
                # Loop through all predictions with the same ic_id (regardless of traj_id or pred_idx)
                for pred_traj_id in self._predictions[ic_id]:
                    for pred_idx, prediction in self._predictions[ic_id][
                        pred_traj_id
                    ].items():
                        sq_err = (
                            (
                                prediction["square_xy_positions"]
                                - gt["square_xy_positions"]
                            )
                            .abs()
                            .float()
                            .mean()
                        )
                        cir_err = (
                            (
                                prediction["circle_xy_positions"]
                                - gt["circle_xy_positions"]
                            )
                            .abs()
                            .float()
                            .mean()
                        )
                        error = 0.5 * (sq_err + cir_err)
                        best_error = (
                            error
                            if best_error is None or error < best_error
                            else best_error
                        )
                ic_errors[traj_id] = best_error.item()
            errors[ic_id] = ic_errors
        return errors

    def compute_metrics(self):
        """Compute per-initial-condition L1 metrics for each stored trajectory."""
        prediction_precision_errors = self.compute_precision()
        ground_truth_recall_errors = self.compute_recall()
        prediction_precision_error = (
            torch.tensor(
                list(prediction_precision_errors.values()), dtype=torch.float32
            )
            .mean()
            .item()
        )
        ground_truth_recall_error = (
            torch.tensor(
                [
                    err
                    for ic_errors in ground_truth_recall_errors.values()
                    for err in ic_errors.values()
                ],
                dtype=torch.float32,
            )
            .mean()
            .item()
        )
        denom = prediction_precision_error + ground_truth_recall_error
        f1_error = 2 * prediction_precision_error * ground_truth_recall_error / denom
        return {
            "prediction_precision_error": prediction_precision_error,
            "ground_truth_recall_error": ground_truth_recall_error,
            "f1_error": f1_error,
            "prediction_precision_errors": prediction_precision_errors,
            "ground_truth_recall_errors": ground_truth_recall_errors,
        }

    def synchronize_across_processes(self):
        state = {
            "ground_truths": dict(self._ground_truths),
            "predictions": {
                ic: {traj: dict(preds) for traj, preds in trajs.items()}
                for ic, trajs in self._predictions.items()
            },
        }

        # num_initial_conditions = len(state["ground_truths"].keys())

        # num_trajectories = sum(len(trajs) for trajs in state["ground_truths"].values())

        # num_predictions = sum(sum(len(preds) for preds in trajs.values())
        #                      for trajs in self._predictions.values())

        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, state)

        merged_gt = defaultdict(dict)
        merged_pred = defaultdict(lambda: defaultdict(dict))

        for item in gathered:
            for ic_id, trajectories in item["ground_truths"].items():
                for traj_id, data in trajectories.items():
                    if traj_id in merged_gt[ic_id]:
                        raise ValueError(
                            f"Duplicate trajectory id '{traj_id}' for context '{ic_id}' across ranks."
                        )
                    merged_gt[ic_id][traj_id] = data

            for ic_id, trajs in item["predictions"].items():
                for traj_id, preds in trajs.items():
                    for pred_idx, data in preds.items():
                        merged_pred[ic_id][traj_id][pred_idx] = data

        self._ground_truths = merged_gt
        self._predictions = merged_pred


class FlowMatchingEvaluatorWaymoFastV2(object):
    """
    Optimized evaluator that runs COCO evaluation once with all predictions.
    Uses RankedCOCOeval to handle prediction_id and video-level ranking.
    """

    def __init__(self, cfg, validate_mini_set=False):
        self.cfg = cfg
        self.validate_mini_set = validate_mini_set

        self.root = cfg.datasets.val.waymo.root
        df_metadata_path = os.path.join(self.root, "df_metadata.csv")
        self.df_metadata = pd.read_csv(df_metadata_path)

        self.number_context_latents = cfg.datasets.val.waymo.number_context_latents
        self.number_target_latents = cfg.datasets.val.waymo.number_target_latents

        self.det_cfg = cfg.get("evaluation_detection", None)
        if self.det_cfg is not None:
            if self.validate_mini_set:
                det_annotations_path = self.det_cfg.path_annotations_eval_mini
            else:
                det_annotations_path = self.det_cfg.path_annotations_eval
            self.coco_gt = COCO(det_annotations_path)
        else:
            self.coco_gt = None

        # Storage for predictions
        self.list_coco_predictions = []
        self.list_coco_predictions_oracle = []

    def _build_clip_id_to_imgs_mapping(self, image_ids):
        """Build mapping from video_id/clip_id to list of image_ids."""
        if self.coco_gt is None:
            return {}
        from collections import defaultdict

        video_to_imgs = defaultdict(list)
        for img_id in image_ids:
            img_info = self.coco_gt.imgs[img_id]
            # Try both field names depending on your annotation format
            clip_id = img_info["clip_id"]
            if clip_id is not None:
                video_to_imgs[clip_id].append(img_id)
            else:
                raise ValueError(
                    f"Image ID {img_id} does not have 'clip_id' in its info."
                )
        return dict(video_to_imgs)

    def accumulate(self, batch, model_outputs, downstream_outputs):
        """Accumulate predictions from a batch."""
        coco_predictions = downstream_outputs["coco_predictions"]
        oracle_predictions = downstream_outputs.get("coco_predictions_oracle", [])

        self.list_coco_predictions.extend(coco_predictions)
        self.list_coco_predictions_oracle.extend(oracle_predictions)

    def _evaluate_with_ranking(
        self, predictions_list, n_predictions_for_evaluation, list_ids_to_evaluate=None
    ):
        if not predictions_list or self.coco_gt is None:
            return {}
        logger = logging.getLogger(__name__)

        # Get all the image ids from the predictions list
        list_img_ids_from_predictions_list = []
        for p in predictions_list:
            list_img_ids_from_predictions_list.append(p["image_id"])
        list_img_ids_from_predictions_list = set(
            sorted(list_img_ids_from_predictions_list)
        )

        logger.info(
            f"Predictions cover {len(list_img_ids_from_predictions_list)} unique image IDs."
        )

        logger.info(
            "Filtering to %d target images (out of %d total)",
            len(list_ids_to_evaluate),
            len(self.coco_gt.imgs),
        )

        predictions_list = [
            p for p in predictions_list if p["image_id"] in list_ids_to_evaluate
        ]

        # 1. Load predictions into COCO format
        coco_dt = self.coco_gt.loadRes(predictions_list)

        # 2. Create evaluator - USE RankedCOCOeval instead of COCOeval
        coco_eval = RankedCOCOeval(self.coco_gt, coco_dt, "bbox")

        # ADD: Set filtered image IDs in params
        coco_eval.params.imgIds = list_ids_to_evaluate

        # 3. Run evaluation ONCE (computes all matches for all predictions)
        logger.info(
            "Running COCO evaluate() - computing matches for all predictions..."
        )
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()

        # ADD: Build video mapping BEFORE ranking
        # To replace by the one of the dataset
        video_to_imgs = self._build_clip_id_to_imgs_mapping(list_ids_to_evaluate)

        # 4. Rank predictions per video
        logger.info("Ranking predictions per video...")
        video_rankings = coco_eval.accumulate_per_video(video_to_imgs)

        # 5. Accumulate and summarize for each ranking type
        metrics = {}
        for ranking_type in ["worst", "median", "best"]:
            logger.info("Computing metrics for %s predictions...", ranking_type)

            # Accumulate with filtered predictions
            coco_eval.accumulate_ranked(
                video_rankings, video_to_imgs, ranking_type=ranking_type
            )

            # Summarize
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval.summarize()

            # Extract metrics
            prefix = ranking_type.capitalize()
            metrics[f"{prefix}_AP"] = coco_eval.stats[0] * 100
            metrics[f"{prefix}_AP_50"] = coco_eval.stats[1] * 100
            metrics[f"{prefix}_AP_75"] = coco_eval.stats[2] * 100
            metrics[f"{prefix}_AP_small"] = coco_eval.stats[3] * 100
            metrics[f"{prefix}_AP_medium"] = coco_eval.stats[4] * 100
            metrics[f"{prefix}_AP_large"] = coco_eval.stats[5] * 100
            metrics[f"{prefix}_AR_1"] = coco_eval.stats[6] * 100
            metrics[f"{prefix}_AR_10"] = coco_eval.stats[7] * 100
            metrics[f"{prefix}_AR_100"] = coco_eval.stats[8] * 100

        return metrics

    def _evaluate_oracle(self, predictions_list, list_ids_to_evaluate=None):
        """Standard evaluation for oracle predictions (no ranking)."""
        if self.coco_gt is None:
            return {}
        # ADD: Filter to target image IDs
        predictions_list = [
            p for p in predictions_list if p["image_id"] in list_ids_to_evaluate
        ]

        if not predictions_list:
            return {}

        coco_dt = self.coco_gt.loadRes(predictions_list)
        from pycocotools.cocoeval import COCOeval

        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")

        # ADD: Set filtered image IDs
        coco_eval.params.imgIds = list_ids_to_evaluate

        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        return {
            "Oracle_AP": coco_eval.stats[0] * 100,
            "Oracle_AP_50": coco_eval.stats[1] * 100,
            "Oracle_AP_75": coco_eval.stats[2] * 100,
            "Oracle_AP_medium": coco_eval.stats[4] * 100,
            "Oracle_AP_large": coco_eval.stats[5] * 100,
        }

    def _log_metrics(self, logger, metrics, prefix):
        """Log metrics in a nice format."""
        ap = metrics.get(f"{prefix}_AP")
        if ap is None:
            return

        logger.info("=" * 60)
        logger.info(f"{prefix} Evaluation Results:")
        logger.info("=" * 60)
        logger.info(f"  AP: {ap:.4f}")
        logger.info(f"  AP@50: {metrics.get(f'{prefix}_AP_50', 0):.4f}")
        logger.info(f"  AP@75: {metrics.get(f'{prefix}_AP_75', 0):.4f}")
        logger.info(f"  AP medium: {metrics.get(f'{prefix}_AP_medium', 0):.4f}")
        logger.info(f"  AP large: {metrics.get(f'{prefix}_AP_large', 0):.4f}")
        logger.info("=" * 60)

    def _deduplicate_by_rank(self, predictions_list):
        """
        Remove duplicate predictions caused by DistributedSampler padding.
        For each (image_id, prediction_id), keeps only predictions from the lowest rank.

        Args:
            predictions_list: List of prediction dicts, each containing 'image_id',
                            'prediction_id', and 'rank'

        Returns:
            Deduplicated list of predictions.
        """
        from collections import defaultdict

        # Group predictions by (image_id, prediction_id)
        grouped = defaultdict(list)
        for pred in predictions_list:
            key = (pred["image_id"], pred.get("prediction_id", 0))
            # assert that 'rank' is in pred
            assert "rank" in pred, "'rank' key missing in prediction dict."
            grouped[key].append(pred)

        # For each group, find minimum rank and keep only those predictions
        deduplicated = []
        for key, preds in grouped.items():
            ranks = [p["rank"] for p in preds]
            min_rank = min(ranks)
            # Keep all predictions from the minimum rank
            deduplicated.extend([p for p in preds if p["rank"] == min_rank])

        return deduplicated

    def evaluate(
        self, logger, n_predictions_for_evaluation=3, list_ids_to_evaluate=None
    ):
        """
        Main evaluation method.

        Args:
            logger: Logger object
            n_predictions_for_evaluation: Number of stochastic predictions per video
            list_ids_to_evaluate: Optional list of image IDs to evaluate on
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Gather predictions from all ranks
        gathered_main = [None for _ in range(world_size)]
        gathered_oracle = [None for _ in range(world_size)]
        dist.gather_object(
            self.list_coco_predictions, gathered_main if rank == 0 else None, dst=0
        )
        dist.gather_object(
            self.list_coco_predictions_oracle,
            gathered_oracle if rank == 0 else None,
            dst=0,
        )

        eval_results = {}

        if rank == 0:
            # Merge predictions
            merged_main = [p for sub in gathered_main for p in sub]
            merged_oracle = [p for sub in gathered_oracle for p in sub]

            # DEDUPLICATE MAIN PREDICTIONS
            logger.info("=" * 80)
            logger.info("DEDUPLICATION - Main Predictions")
            logger.info("=" * 80)
            main_before = len(merged_main)
            logger.info(f"Before deduplication: {main_before} predictions")

            merged_main = self._deduplicate_by_rank(merged_main)

            main_after = len(merged_main)
            main_removed = main_before - main_after
            main_removed_pct = (
                (main_removed / main_before * 100) if main_before else 0.0
            )
            logger.info(f"After deduplication: {main_after} predictions")
            logger.info(
                f"Difference: {main_removed} predictions ({main_removed_pct:.2f}%)"
            )

            # DEDUPLICATE ORACLE PREDICTIONS
            logger.info("=" * 80)
            logger.info("DEDUPLICATION - Oracle Predictions")
            logger.info("=" * 80)
            oracle_before = len(merged_oracle)
            logger.info(f"Before deduplication: {oracle_before} predictions")

            merged_oracle = self._deduplicate_by_rank(merged_oracle)

            oracle_after = len(merged_oracle)
            oracle_removed = oracle_before - oracle_after
            oracle_removed_pct = (
                (oracle_removed / oracle_before * 100) if oracle_before else 0.0
            )
            logger.info(f"After deduplication: {oracle_after} predictions")
            logger.info(
                f"Difference: {oracle_removed} predictions ({oracle_removed_pct:.2f}%)"
            )
            logger.info("=" * 80)

            # Evaluate main predictions (with ranking)
            if merged_main:
                if "prediction_id" in merged_main[0]:
                    logger.info("=" * 80)
                    logger.info(
                        "EVALUATING STOCHASTIC PREDICTIONS (with video-level ranking)"
                    )
                    logger.info("=" * 80)
                    main_metrics = self._evaluate_with_ranking(
                        merged_main,
                        n_predictions_for_evaluation,
                        list_ids_to_evaluate=list_ids_to_evaluate,
                    )
                    eval_results.update(main_metrics)

                    # Log results
                    for prefix in ["Worst", "Median", "Best"]:
                        self._log_metrics(logger, main_metrics, prefix)

            # Evaluate oracle predictions
            if merged_oracle:
                logger.info("=" * 80)
                logger.info("EVALUATING ORACLE PREDICTIONS")
                logger.info("=" * 80)
                oracle_metrics = self._evaluate_oracle(
                    merged_oracle, list_ids_to_evaluate=list_ids_to_evaluate
                )
                eval_results.update(oracle_metrics)
                self._log_metrics(logger, oracle_metrics, "Oracle")

        # Synchronize
        if dist.is_initialized():
            dist.barrier()

        # Clear predictions
        self.list_coco_predictions.clear()
        self.list_coco_predictions_oracle.clear()

        # Format for return
        final_metrics = {f"coco/{k.lower()}": v for k, v in eval_results.items()}
        return final_metrics


class PushTEvaluator:
    """Minimal evaluator for PushT — just tracks L1 latent losses (no COCO)."""

    def __init__(self, cfg=None, **kwargs):
        pass

    def accumulate(self, batch, model_outputs, downstream_outputs=None):
        pass

    def evaluate(
        self, logger=None, n_predictions_for_evaluation=1, list_ids_to_evaluate=None
    ):
        return {}
