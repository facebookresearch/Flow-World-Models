import torch as th
import numpy as np

import enum

from . import path
from .utils import mean_flat
from .integrators import ode, sde
from losses import GramAnchoringLoss


class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)


class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


def truncated_logitnormal_sample(shape, mu, sigma, low=0.0, high=1.0):
    """
    Samples X in (0,1) with Z = logit(X) ~ Normal(mu, sigma^2), truncated so X in [low, high].
    Works for scalars or tensors mu/sigma/low/high with broadcasting.

    Args:
        shape: output batch shape (e.g., (N,) or (N,M)). Leave () to broadcast to mu.shape.
        mu, sigma: tensors or floats (sigma > 0).
        low, high: truncation bounds in [0,1]. (low can be 0, high can be 1).
        device, dtype: optional overrides.

    Returns:
        Tensor of samples with shape = broadcast(shape, mu.shape, ...)
    """
    mu = th.as_tensor(mu)
    sigma = th.as_tensor(sigma)
    low = th.as_tensor(low)
    high = th.as_tensor(high)

    # Map truncation bounds to logit space; handles 0/1 → ±inf automatically.
    z_low = th.logit(low)  # = -inf if low==0
    z_high = th.logit(high)  # = +inf if high==1

    # Standardize bounds for the base Normal(0,1)
    base = th.distributions.Normal(th.zeros_like(mu), th.ones_like(sigma))
    alpha = (z_low - mu) / sigma
    beta = (z_high - mu) / sigma

    # Truncated-normal inverse CDF sampling:
    # U ~ Uniform(Φ(alpha), Φ(beta));  Z = mu + sigma * Φ^{-1}(U);  X = sigmoid(Z)
    cdf_alpha = base.cdf(alpha)
    cdf_beta = base.cdf(beta)

    # Draw uniforms on the truncated interval
    out_shape = th.broadcast_shapes(shape, mu.shape, sigma.shape, low.shape, high.shape)
    U = th.rand(out_shape, device=mu.device, dtype=mu.dtype)
    U = cdf_alpha + (cdf_beta - cdf_alpha) * U.clamp_(0, 1)

    Z = mu + sigma * base.icdf(U)
    X = th.sigmoid(Z)

    # Numerical safety when low/high are extremely close; clamp back into [low, high].
    return X.clamp(low, high)


class Transport:
    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        time_dist_type,
        time_dist_shift,
        train_eps,
        sample_eps,
        loss_norm,
        consistency_loss,
        gram_anchoring_loss,
        noise_std=1.0,
        consistency_loss_use_direct_delta_target=False,
        predict_x_derive_velocity=False,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        assert self.time_dist_shift >= 1.0, "time distribution shift must be >= 1.0."
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.loss_norm = loss_norm
        if self.loss_norm not in {"l1", "l2"}:
            raise ValueError(
                f"Unsupported loss_norm '{loss_norm}'. Expected 'l1' or 'l2'."
            )

        # Consistency loss
        self.consistency_loss_enabled = consistency_loss["enabled"]
        if self.consistency_loss_enabled:
            self.consistency_loss_weight = consistency_loss["weight"]
            self.consistency_loss_type = consistency_loss["type"]
            self.consistency_loss_order = consistency_loss["order"]
            self.enforce_context_consistency = consistency_loss[
                "enforce_context_consistency"
            ]

        # Gram anchoring loss
        self.gram_anchoring_loss_enabled = gram_anchoring_loss["enabled"]
        if self.gram_anchoring_loss_enabled:
            self.gram_anchoring_loss = GramAnchoringLoss(
                weight_loss=gram_anchoring_loss["weight_loss"],
                loss_type=gram_anchoring_loss["loss_type"],
                temporal=gram_anchoring_loss["temporal"],
                time_weighting=gram_anchoring_loss["time_weighting"],
                tau_min=gram_anchoring_loss["tau_min"],
                tau_max=gram_anchoring_loss["tau_max"],
            )

        self.noise_std = noise_std
        self.consistency_loss_use_direct_delta_target = (
            consistency_loss_use_direct_delta_target
        )
        self.predict_x_derive_velocity = predict_x_derive_velocity
        self.t_eps = 5e-2

    def prior_logp(self, z):
        """
        Standard multivariate normal prior
        Assume z is batched
        """
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2.0 * np.log(2 * np.pi) - th.sum(x**2) / 2.0
        return th.vmap(_fn)(z)

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1 - 1 / 1000
        eps = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [path.VPCPlan]:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) and (
            self.model_type != ModelType.VELOCITY or sde
        ):  # avoid numerical issue by taking a first semi-implicit step
            t0 = (
                eps
                if (diffusion_form == "SBDM" and sde)
                or self.model_type != ModelType.VELOCITY
                else 0
            )
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
        Args:
          x1 - data point; [batch, *dim]
        """
        x0 = th.randn_like(x1)
        x0 = x0 * self.noise_std
        dist_options = self.time_dist_type.split("_")
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        if dist_options[0] == "uniform":
            t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        elif dist_options[0] == "logit-normal":
            assert len(dist_options) == 3, (
                "Logit-normal distribution must specify the mean and variance."
            )
            mu, sigma = float(dist_options[1]), float(dist_options[2])
            assert sigma > 0, "Logit-normal distribution must have positive variance."
            t = truncated_logitnormal_sample(
                (x1.shape[0],), mu=mu, sigma=sigma, low=t0, high=t1
            )
        else:
            raise NotImplementedError(
                f"Unknown time distribution type {self.time_dist_type}"
            )

        t = t.to(x1)
        # sqrt_size_ratio = 1 / self.time_dist_shift # already sqrted
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t, x0, x1

    def _apply_loss_norm(self, tensor):
        if self.loss_norm == "l1":
            return tensor.abs()
        elif self.loss_norm == "l2":
            return tensor**2
        else:
            raise ValueError(
                f"Unsupported loss_norm '{self.loss_norm}'. Expected 'l1' or 'l2'."
            )

    def _compute_temporal_diff(self, x, model_kwargs=None, negate=False):
        """
        Compute temporal finite differences
        x: [B, D, T, H, W]
        negate: if True, return the negation of the difference (for correlated noise case)
        """
        sign = -1.0 if negate else 1.0

        if self.enforce_context_consistency:
            context_latents = model_kwargs["context_latents"]
            last_frame_context_latents = context_latents[
                :, :, -1:, :, :
            ]  # [B, D, 1, H, W]
            x = th.cat([last_frame_context_latents, x], dim=2)  # [B, D, T+1, H, W]

        if self.consistency_loss_order == 1:
            # First order: x[t+1] - x[t]
            return sign * (x[:, :, 1:] - x[:, :, :-1])  # [B, D, T-1, H, W]
        elif self.consistency_loss_order == 2:
            # Second order: x[t+1] - 2*x[t] + x[t-1] (acceleration/curvature)
            return sign * (
                x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]
            )  # [B, D, T-2, H, W]
        else:
            raise ValueError(f"Unsupported order {self.consistency_loss_order}")

    def _apply_consistency_loss_norm(self, tensor):
        """Apply loss norm for consistency loss (may differ from main loss)."""
        if self.consistency_loss_type == "l1":
            return tensor.abs()
        elif self.consistency_loss_type == "l2":
            return tensor**2
        else:
            raise ValueError(
                f"Unsupported consistency loss type '{self.consistency_loss_type}'"
            )

    def training_losses(self, model, x1, model_kwargs=None):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint, shape [B, D, T, H, W]
        - model_kwargs: additional arguments for the model. Typically used for conditional generation
        """
        if model_kwargs == None:
            model_kwargs = {}

        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        model_output = model(x=xt, t=t, **model_kwargs)
        # convert model output in fp32
        model_output = model_output.float()
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        terms = {}
        if self.model_type == ModelType.VELOCITY:
            if self.predict_x_derive_velocity:
                # Model predicts x1 (clean data), derive velocity
                # In reversed convention: v = (xt - x1) / t
                x1_pred = model_output
                t_expanded = path.expand_t_like_x(t, xt)
                # Clamp t to avoid division by zero near t=0
                v_pred = (xt - x1_pred) / t_expanded.clamp_min(self.t_eps)

                # Loss on velocity
                loss_flow = mean_flat(self._apply_loss_norm(v_pred - ut))

                # Store for potential use in consistency loss
                terms["x1_pred"] = x1_pred
                terms["v_pred"] = v_pred
            else:
                # Original: model directly predicts velocity
                loss_flow = mean_flat(self._apply_loss_norm(model_output - ut))
                # Compute x1_pred via one-step projection
                t_expanded = path.expand_t_like_x(t, xt)
                x1_pred = xt - t_expanded * model_output  # [B, D, T, H, W]
                terms["x1_pred"] = x1_pred
                terms["v_pred"] = model_output
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t**2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                loss_flow = mean_flat(weight * self._apply_loss_norm(model_output - x0))
            else:
                loss_flow = mean_flat(
                    weight * self._apply_loss_norm(model_output * sigma_t + x0)
                )

        terms["loss_flow"] = loss_flow
        terms["loss"] = loss_flow
        terms["t"] = t.detach()

        if self.consistency_loss_enabled:
            if self.model_type == ModelType.VELOCITY:
                if self.consistency_loss_use_direct_delta_target:
                    diff_pred = self._compute_temporal_diff(
                        model_output, model_kwargs, negate=False
                    )
                    diff_gt = self._compute_temporal_diff(
                        ut, model_kwargs, negate=False
                    )
                else:
                    # Use the already-computed x1_pred
                    diff_pred = self._compute_temporal_diff(
                        terms["x1_pred"], model_kwargs, negate=False
                    )
                    diff_gt = self._compute_temporal_diff(
                        x1, model_kwargs, negate=False
                    )
            else:
                raise ValueError(
                    "Consistency loss is only implemented for velocity model type"
                )

            loss_consistency = mean_flat(
                self._apply_consistency_loss_norm(diff_pred - diff_gt)
            )
            terms["loss_consistency"] = loss_consistency
            terms["loss"] = (
                terms["loss"] + self.consistency_loss_weight * loss_consistency
            )

        if self.gram_anchoring_loss_enabled:
            if self.model_type == ModelType.VELOCITY:
                gram_batch = {"target_latents": x1}
                gram_outputs = {"predicted_latents": terms["x1_pred"]}
                gram_loss_dict = self.gram_anchoring_loss.compute_loss(
                    gram_batch,
                    gram_outputs,
                    normalize_d=True,
                    t=t,
                )
                terms["loss_gram"] = gram_loss_dict["loss_gram"]
                terms["loss"] = terms["loss"] + gram_loss_dict["loss_gram"]
            else:
                raise ValueError(
                    "Gram anchoring loss is only implemented for velocity model type"
                )

        return terms

    def get_drift(self):
        """member function for obtaining the drift of the probability flow ODE"""

        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output  # by change of variable

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        def velocity_from_x_ode(x, t, model, **model_kwargs):
            x1_pred = model(x, t, **model_kwargs)
            t_expanded = path.expand_t_like_x(t, x)
            velocity = (x - x1_pred) / t_expanded.clamp_min(self.t_eps)
            return velocity

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        elif self.model_type == ModelType.VELOCITY:
            if self.predict_x_derive_velocity:
                drift_fn = velocity_from_x_ode
            else:
                drift_fn = velocity_ode
        else:
            raise NotImplementedError()

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, (
                "Output shape from ODE solver must match input shape"
            )
            return model_output

        return body_fn

    def get_score(
        self,
    ):
        """member function for obtaining score of
        x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: (
                model(x, t, **kwargs)
                / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
            )
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: (
                self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
            )
        else:
            raise NotImplementedError()

        return score_fn


class Sampler:
    """Sampler class for the transport model"""

    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """

        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def sde_diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(
                x, t, form=diffusion_form, norm=diffusion_norm
            )
            return diffusion

        def sde_drift_fn(x, t, model, **kwargs):
            drift_mean = self.drift(x, t, model, **kwargs) - sde_diffusion_fn(
                x, t
            ) * self.score(x, t, model, **kwargs)
            return drift_mean

        return sde_drift_fn, sde_diffusion_fn

    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""

        if last_step is None:
            last_step_fn = lambda x, t, model, **model_kwargs: x
        elif last_step == "Mean":
            last_step_fn = lambda x, t, model, **model_kwargs: (
                x - sde_drift(x, t, model, **model_kwargs) * last_step_size
            )
        elif last_step == "Tweedie":
            alpha = (
                self.transport.path_sampler.compute_alpha_t
            )  # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = lambda x, t, model, **model_kwargs: (
                x / alpha(t)[0][0]
                + (sigma(t)[0][0] ** 2)
                / alpha(t)[0][0]
                * self.score(x, t, model, **model_kwargs)
            )
        elif last_step == "Euler":
            last_step_fn = lambda x, t, model, **model_kwargs: (
                x - self.drift(x, t, model, **model_kwargs) * last_step_size
            )
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
            time_dist_shift=self.transport.time_dist_shift,
        )

        last_step_fn = self.__get_last_step(
            sde_drift, last_step=last_step, last_step_size=last_step_size
        )

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * (1 - t1)
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(
                x, th.ones_like(t) * (1 - t), model, **kwargs
            )
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            time_dist_shift=self.transport.time_dist_shift,
        )
        return _ode.sample
