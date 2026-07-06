import os
from time import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from trainer.trainer import Trainer
from trainer.utils import (
    _gather_tensor_across_ranks,
    _move_batch_to_device,
)
from trainer.utils_latents import extract_backbone_latents
from utils.utils_format import format_loss_dict


class TrainerImageReconstruction(Trainer):
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
        self.loss_cls = (
            self.loss_cls.to(self.device)
            if hasattr(self.loss_cls, "to")
            else self.loss_cls
        )

        training_cfg = cfg.trainer.training
        eval_cfg = cfg.trainer.evaluation
        self.epochs = training_cfg.epochs
        self.eval_every = eval_cfg.eval_every
        self.eval_first = eval_cfg.eval_first
        self.checkpoint_every = cfg.trainer.checkpoint_every
        self.gradient_clip_enabled = training_cfg.gradient_clipping
        self.gradient_clip_max_norm = training_cfg.gradient_clipping_max_norm
        self.save_num_images = int(eval_cfg.save_num_images)
        self.do_final_full_evaluation = bool(eval_cfg.do_final_full_evaluation)
        self.ema_model = None
        self.model_name = cfg.model.model_name

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

            if (
                not self.args.debug
                and self.train_state.epoch % self.checkpoint_every == 0
            ):
                self.save_checkpoint()

            if self.args.debug:
                self.run_evaluation(final_eval=False)
            elif self.train_state.epoch % self.eval_every == 0:
                self.run_evaluation(final_eval=False)

        if self.do_final_full_evaluation:
            self.run_evaluation(final_eval=True)

    def _extract_image_latents(self, backbone_images):
        latents, _ = extract_backbone_latents(
            self.preprocessor,
            self.backbone,
            self.backbone_name,
            backbone_images.unsqueeze(1),
            self.device,
            flatten_temporal_dim=False,
            average_intermediate_layers=self.average_intermediate_layers,
            use_dinov3_layer_norm=self.use_dinov3_layer_norm,
            extract_single_layer=self.extract_single_layer,
            autocast_dtype=self.autocast_dtype,
        )
        latents = latents.squeeze(2)
        return latents

    @staticmethod
    def _rescale_to_unit_interval(images):
        return images.clamp(-1.0, 1.0).add(1.0).mul(0.5)

    def _save_reconstruction_grid(
        self, targets, predictions, dump_epoch_dir, batch_idx
    ):
        if self.global_rank != 0:
            return
        num_images = min(self.save_num_images, targets.shape[0])
        if num_images == 0:
            return
        comparison = torch.cat(
            [targets[:num_images], predictions[:num_images]],
            dim=0,
        )
        comparison = self._rescale_to_unit_interval(comparison.cpu())
        output_path = os.path.join(dump_epoch_dir, f"recon_batch_{batch_idx:04d}.png")
        save_image(comparison, output_path, nrow=num_images)

    def train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            image_latents = self._extract_image_latents(batch["backbone_image"])

        with self._autocast_context():
            model_outputs = self.model(image_latents)
            loss_dict = self.loss_cls.compute_loss(batch, model_outputs)

        total_loss = loss_dict["loss_total"]
        self.scaler.scale(total_loss).backward()
        if self.gradient_clip_enabled:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_max_norm
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss_dict

    def train_epoch(self, epoch_idx):
        self.model.train()
        self.epoch_running_losses = {}
        self.log_losses = {}
        num_batches = 0
        epoch_start_time = time()

        for batch_idx, batch in enumerate(self.train_loader):
            if self.args.debug and batch_idx >= 10:
                self.logger.info("Debug mode: breaking after 10 train batches")
                break

            num_batches += 1
            batch = _move_batch_to_device(batch, self.device)
            loss_dict = self.train_step(batch)
            self.accumulate_running_losses(self.log_losses, loss_dict)
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
                    self.epoch_running_losses, self.log_losses
                )
                self.log_losses = {}

        if self.log_losses:
            self.accumulate_running_losses(self.epoch_running_losses, self.log_losses)

        if num_batches == 0:
            return
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
        dump_epoch_dir = self._setup_eval_dirs(final_eval)
        self.epoch_running_losses = {}
        num_batches = 0
        compute_fid_this_eval = bool(
            self.evaluator is not None and getattr(self.evaluator, "compute_fid", False)
        )

        if self.evaluator is not None and self.global_rank == 0:
            self.evaluator.reset(include_fid=compute_fid_this_eval)

        if final_eval and not eval_only_use_mini:
            eval_loader = self.eval_loader
        else:
            eval_loader = self.eval_mini_loader

        if compute_fid_this_eval:
            self._ensure_cached_real_fid_stats(eval_loader)
        if dist.is_initialized():
            dist.barrier()

        for batch_idx, batch in enumerate(eval_loader):
            if self.args.debug and batch_idx >= 3:
                self.logger.info("Debug mode: breaking after 3 eval batches")
                break

            num_batches += 1
            batch = _move_batch_to_device(batch, self.device)
            image_latents = self._extract_image_latents(batch["backbone_image"])

            with self._autocast_context():
                model_outputs = self.model(image_latents)
                loss_dict = self.loss_cls.compute_loss(batch, model_outputs)
            self.accumulate_running_losses(self.epoch_running_losses, loss_dict)

            gathered_targets = _gather_tensor_across_ranks(
                batch["image_normalized"], self.global_rank
            )
            gathered_predictions = _gather_tensor_across_ranks(
                model_outputs["reconstructed_images"], self.global_rank
            )
            if self.global_rank == 0 and self.evaluator is not None:
                self.evaluator.update(
                    gathered_targets,
                    gathered_predictions,
                    update_fid=compute_fid_this_eval,
                )
                if batch_idx < 2:
                    self._save_reconstruction_grid(
                        gathered_targets,
                        gathered_predictions,
                        dump_epoch_dir,
                        batch_idx,
                    )

        if num_batches == 0:
            return

        averaged_losses = self._compute_averaged_losses(
            num_batches, self.epoch_running_losses
        )
        prefix = "final_eval/" if final_eval else "eval/"
        formatted_losses = format_loss_dict(averaged_losses, prefix=prefix)
        eval_metrics = {}
        if self.global_rank == 0 and self.evaluator is not None:
            eval_metrics = format_loss_dict(
                self.evaluator.compute(include_fid=compute_fid_this_eval),
                prefix=prefix,
            )

        if dist.is_initialized():
            dist.barrier()

        if self.global_rank == 0:
            summary = {}
            summary.update(formatted_losses)
            summary.update(eval_metrics)
            summary[f"{prefix}duration_sec"] = time() - eval_start_time
            if self.wandb_run is not None:
                self.wandb_run.log(summary, step=self.train_state.gradient_steps)
            self.logger.info(
                "[EVAL] epoch %d | l1=%.4f | psnr=%.3f | fid=%s | duration=%.3fs",
                self.train_state.epoch,
                summary.get(f"{prefix}l1", float("nan")),
                summary.get(f"{prefix}psnr", float("nan")),
                f"{summary.get(f'{prefix}fid', float('nan')):.3f}"
                if f"{prefix}fid" in summary
                else "n/a",
                summary[f"{prefix}duration_sec"],
            )

    @torch.inference_mode()
    def _ensure_cached_real_fid_stats(self, eval_loader):
        if (
            self.global_rank != 0
            or self.evaluator is None
            or not self.evaluator.compute_fid
        ):
            return
        cache_loader = DataLoader(
            dataset=eval_loader.dataset,
            batch_size=self.cfg.trainer.evaluation.batch_size_per_gpu,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=self.args.num_workers > 0,
            collate_fn=eval_loader.collate_fn,
        )
        self.evaluator.ensure_real_features(cache_loader, self.device)
