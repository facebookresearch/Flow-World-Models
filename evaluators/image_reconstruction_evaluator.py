import logging
import math
from pathlib import Path

import torch


logger = logging.getLogger(__name__)


class ImageReconstructionEvaluator:
    def __init__(self, cfg):
        eval_cfg = cfg.trainer.evaluation
        self.compute_fid = bool(eval_cfg.compute_fid)
        self.compute_ssim = bool(eval_cfg.compute_ssim)
        self.fid_real_stats_path = (
            Path(eval_cfg.fid_real_stats_path) if self.compute_fid else None
        )
        self.real_features_ready = False

        self.fid_metric = None
        if self.compute_fid:
            from torchmetrics.image.fid import FrechetInceptionDistance

            self.fid_metric = FrechetInceptionDistance(
                normalize=False, reset_real_features=False
            )

        self.ssim_metric = None
        if self.compute_ssim:
            from torchmetrics.image import StructuralSimilarityIndexMeasure

            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0)

        self.reset()

    def reset(self, include_fid: bool = True):
        self.num_images = 0
        self.l1_sum = 0.0
        self.mse_sum = 0.0
        self.num_values = 0
        if self.fid_metric is not None and include_fid:
            self.fid_metric.reset()
        if self.ssim_metric is not None:
            self.ssim_metric.reset()

    def _load_cached_real_features(self, device):
        cache = torch.load(self.fid_real_stats_path, map_location=device)
        required_keys = {
            "real_features_sum",
            "real_features_cov_sum",
            "real_features_num_samples",
        }
        if not isinstance(cache, dict) or not required_keys.issubset(cache.keys()):
            return False

        self.fid_metric.real_features_sum = cache["real_features_sum"].to(
            device=device, dtype=torch.float64
        )
        self.fid_metric.real_features_cov_sum = cache["real_features_cov_sum"].to(
            device=device,
            dtype=torch.float64,
        )
        self.fid_metric.real_features_num_samples = cache[
            "real_features_num_samples"
        ].to(
            device=device,
        )
        self.real_features_ready = (
            int(self.fid_metric.real_features_num_samples.item()) > 1
        )
        return self.real_features_ready

    def _save_cached_real_features(self):
        self.fid_real_stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "real_features_sum": self.fid_metric.real_features_sum.detach().cpu(),
                "real_features_cov_sum": self.fid_metric.real_features_cov_sum.detach().cpu(),
                "real_features_num_samples": self.fid_metric.real_features_num_samples.detach().cpu(),
            },
            self.fid_real_stats_path,
        )

    def ensure_real_features(self, dataloader, device):
        if self.fid_metric is None or self.real_features_ready:
            return

        start_time = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )
        end_time = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )
        wall_start = None
        if start_time is None:
            import time

            wall_start = time.time()
        else:
            start_time.record()

        self.fid_metric = self.fid_metric.to(device)
        if self.fid_real_stats_path.exists():
            logger.info(
                "Loading cached FID real features from %s", self.fid_real_stats_path
            )
            if self._load_cached_real_features(device):
                if end_time is not None:
                    end_time.record()
                    torch.cuda.synchronize(device)
                    logger.info(
                        "Loaded cached FID real features in %.3fs",
                        start_time.elapsed_time(end_time) / 1000.0,
                    )
                else:
                    import time

                    logger.info(
                        "Loaded cached FID real features in %.3fs",
                        time.time() - wall_start,
                    )
                return
            logger.warning(
                "Cached FID file at %s is invalid or from an older format; recomputing real features.",
                self.fid_real_stats_path,
            )

        num_batches = len(dataloader)
        logger.info(
            "Computing FID real features from scratch over %d batches", num_batches
        )
        for batch_idx, batch in enumerate(dataloader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            self.fid_metric.update(self._to_uint8(images), real=True)
            if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == num_batches:
                logger.info(
                    "Caching FID real features: batch %d/%d",
                    batch_idx,
                    num_batches,
                )

        self._save_cached_real_features()
        self.real_features_ready = True
        if end_time is not None:
            end_time.record()
            torch.cuda.synchronize(device)
            logger.info(
                "Computed and cached FID real features in %.3fs",
                start_time.elapsed_time(end_time) / 1000.0,
            )
        else:
            import time

            logger.info(
                "Computed and cached FID real features in %.3fs",
                time.time() - wall_start,
            )

    @staticmethod
    def _to_uint8(images):
        if images.dtype == torch.uint8:
            return images

        images = images.float()
        image_min = float(images.min().item())
        image_max = float(images.max().item())

        if 0.0 <= image_min and image_max <= 255.0:
            return images.round().to(torch.uint8)

        if -1.0 <= image_min and image_max <= 1.0:
            images = images.clamp(-1.0, 1.0).add(1.0).mul(127.5).round()
            return images.to(torch.uint8)

        raise ValueError(
            f"Expected images in uint8/[0,255] or normalized [-1,1], got min={image_min:.3f}, max={image_max:.3f}"
        )

    @staticmethod
    def _matrix_sqrt_psd(matrix: torch.Tensor) -> torch.Tensor:
        matrix = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = eigvals.clamp_min(0.0).sqrt()
        return (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.T

    def _compute_fid_stable(self) -> float:
        real_n = int(self.fid_metric.real_features_num_samples.item())
        fake_n = int(self.fid_metric.fake_features_num_samples.item())
        if real_n < 2 or fake_n < 2:
            raise RuntimeError(
                "More than one sample is required for both real and fake distributions to compute FID"
            )

        device = torch.device("cpu")
        real_sum = self.fid_metric.real_features_sum.to(
            device=device, dtype=torch.float64
        )
        fake_sum = self.fid_metric.fake_features_sum.to(
            device=device, dtype=torch.float64
        )
        real_cov_sum = self.fid_metric.real_features_cov_sum.to(
            device=device, dtype=torch.float64
        )
        fake_cov_sum = self.fid_metric.fake_features_cov_sum.to(
            device=device, dtype=torch.float64
        )

        mean_real = real_sum / real_n
        mean_fake = fake_sum / fake_n
        cov_real = (real_cov_sum - real_n * torch.outer(mean_real, mean_real)) / (
            real_n - 1
        )
        cov_fake = (fake_cov_sum - fake_n * torch.outer(mean_fake, mean_fake)) / (
            fake_n - 1
        )
        cov_real = 0.5 * (cov_real + cov_real.T)
        cov_fake = 0.5 * (cov_fake + cov_fake.T)

        sqrt_cov_real = self._matrix_sqrt_psd(cov_real)
        middle = sqrt_cov_real @ cov_fake @ sqrt_cov_real
        middle = 0.5 * (middle + middle.T)
        trace_sqrt = torch.linalg.eigvalsh(middle).clamp_min(0.0).sqrt().sum()

        mean_diff = (mean_real - mean_fake).square().sum()
        fid = mean_diff + cov_real.trace() + cov_fake.trace() - 2.0 * trace_sqrt
        return float(fid.item())

    def update(self, targets, predictions, update_fid: bool = True):
        batch_size = int(targets.shape[0])
        self.num_images += batch_size
        self.num_values += int(targets.numel())
        self.l1_sum += torch.nn.functional.l1_loss(
            predictions, targets, reduction="sum"
        ).item()
        self.mse_sum += torch.nn.functional.mse_loss(
            predictions, targets, reduction="sum"
        ).item()

        if self.fid_metric is not None and update_fid:
            self.fid_metric = self.fid_metric.to(predictions.device)
            if not self.real_features_ready:
                raise RuntimeError(
                    "FID real features must be cached before calling update()."
                )
            predictions_uint8 = self._to_uint8(predictions)
            self.fid_metric.update(predictions_uint8, real=False)

        if self.ssim_metric is not None:
            self.ssim_metric = self.ssim_metric.to(predictions.device)
            self.ssim_metric.update(predictions, targets)

    def compute(self, include_fid: bool = True):
        if self.num_images == 0:
            return {}

        denom = float(self.num_values)
        l1 = self.l1_sum / denom
        mse = max(self.mse_sum / denom, 1e-12)
        psnr = 10.0 * math.log10(4.0 / mse)

        metrics = {
            "l1": l1,
            "psnr": psnr,
        }
        if self.fid_metric is not None and include_fid:
            metrics["fid"] = self._compute_fid_stable()
        if self.ssim_metric is not None:
            metrics["ssim"] = float(self.ssim_metric.compute().item())
        return metrics
