import torch as th
from torchdiffeq import odeint


def gradient(output, x, grad_outputs=None, create_graph=False, retain_graph=False):
    """
    Compute the gradient of the inner product of output and grad_outputs w.r.t x.
    """
    if grad_outputs is None:
        grad_outputs = th.ones_like(output).detach()
    grad = th.autograd.grad(
        output,
        x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
    )[0]
    return grad


class sde:
    """SDE solver class"""

    def __init__(
        self,
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
        time_dist_shift,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = 1 - th.linspace(t0, t1, num_steps)
        self.t = time_dist_shift * self.t / (1 + (time_dist_shift - 1) * self.t)
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type
        self.time_dist_shift = time_dist_shift

    def __Euler_Maruyama_step(self, x, mean_x, t_curr, t_next, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t_curr
        dw = w_cur * th.sqrt(t_curr - t_next)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x - drift * (t_curr - t_next)
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x

    def __Heun_step(self, x, _, t_curr, t_next, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(t_curr - t_next)
        diffusion = self.diffusion(x, th.ones(x.size(0)).to(x) * t_curr)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, th.ones(x.size(0)).to(x) * t_curr, model, **model_kwargs)
        xp = xhat - (t_curr - t_next) * K1
        K2 = self.drift(xp, th.ones(x.size(0)).to(x) * t_next, model, **model_kwargs)
        return xhat - 0.5 * (t_curr - t_next) * (
            K1 + K2
        ), xhat  # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "euler": self.__Euler_Maruyama_step,
            "heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")

        return sampler

    def sample(self, init, model, **model_kwargs) -> tuple[th.Tensor]:
        """forward loop of sde"""
        x = init
        mean_x = init
        samples = []
        sampler = self.__forward_fn()
        for t_curr, t_next in zip(self.t[:-1], self.t[1:]):
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, t_curr, t_next, model, **model_kwargs)
                samples.append(x)

        return samples


class ode:
    """ODE solver class"""

    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        time_dist_shift,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        # self.t = th.linspace(t0, t1, num_steps)
        self.t = 1 - th.linspace(t0, t1, num_steps)
        self.t = time_dist_shift * self.t / (1 + (time_dist_shift - 1) * self.t)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type
        self.num_steps = num_steps

    def sample(
        self,
        x,
        model,
        compute_likelihood=False,
        log_p_noise=None,
        exact_divergence=False,
        n_acc=10,
        **model_kwargs,
    ):
        """
        Sample from the flow model, optionally computing log-likelihood.

        Args:
            x: Initial noise samples at t=1. Shape [B, ...].
            model: Velocity model callable.
            compute_likelihood: If True, also compute log-likelihood of generated samples.
            log_p_noise: Callable that computes log p(x) for noise distribution.
                         Required if compute_likelihood=True.
            exact_divergence: If True, compute exact divergence (expensive).
                             Default False uses Hutchinson estimator.
            n_acc: Number of accumulations for divergence estimation.
            **model_kwargs: Additional arguments for the model (e.g., context_latents).

        Returns:
            If compute_likelihood=False:
                samples: Tensor of shape [num_steps, B, ...] with trajectory.
            If compute_likelihood=True:
                tuple: (samples, log_likelihood)
                    - samples: Tensor of shape [num_steps, B, ...] with trajectory.
                    - log_likelihood: Tensor of shape [B] with log p(x_data) for each sample.
        """
        device = x[0].device if isinstance(x, tuple) else x.device
        batch_size = x[0].shape[0] if isinstance(x, tuple) else x.shape[0]

        if compute_likelihood:
            assert log_p_noise is not None, (
                "log_p_noise must be provided when compute_likelihood=True"
            )
            return self._sample_with_likelihood(
                x, model, log_p_noise, exact_divergence, n_acc, **model_kwargs
            )
        return self._sample_standard(x, model, **model_kwargs)

    def _sample_standard(self, x, model, **model_kwargs):
        """Standard sampling without likelihood computation."""
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            t = (
                th.ones(x[0].size(0)).to(device) * t
                if isinstance(x, tuple)
                else th.ones(x.size(0)).to(device) * t
            )
            model_output = self.drift(x, t, model, **model_kwargs)
            return model_output

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        samples = odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)
        return samples, None

    def _sample_with_likelihood(
        self,
        x_noise,
        model,
        log_p_noise,
        exact_divergence=False,
        n_acc=10,
        **model_kwargs,
    ):
        """
        Sample while computing log-likelihood using change of variables.

        We integrate from t=1 (noise) to t=0 (data), accumulating divergence.
        The change of variables formula gives:
            log p(x_data) = log p(x_noise) + ∫₁⁰ div(v_t(x_t)) dt

        Args:
            x_noise: Initial noise at t=1. Shape [B, D, T, H, W] or [B, D].
            model: Velocity model callable.
            log_p_noise: Callable for log p(noise).
            exact_divergence: Use exact (expensive) or Hutchinson estimator.
            n_acc: Number of Hutchinson probes per denoising step.
            **model_kwargs: Additional model kwargs.

        Returns:
            tuple: (samples_trajectory, log_likelihood)
        """
        device = x_noise.device
        batch_size = x_noise.shape[0]

        log_p_init = log_p_noise(x_noise)

        def dynamics_with_divergence(t_val, states):
            """
            Joint dynamics for (x_t, accumulated_log_det).
            Returns (velocity, divergence) at current state.
            """
            xt = states[0]

            with th.set_grad_enabled(True):
                xt_grad = xt.detach().requires_grad_(True)

                t_tensor = th.ones(batch_size, device=device) * t_val
                ut = self.drift(xt_grad, t_tensor, model, **model_kwargs)

                if exact_divergence:
                    flat_ut = ut.flatten(start_dim=1)
                    dim = flat_ut.shape[1]
                    div = th.zeros(batch_size, device=device)
                    for i in range(dim):
                        g = gradient(flat_ut[:, i].sum(), xt_grad, create_graph=False)
                        div = div + g.flatten(start_dim=1)[:, i]
                else:
                    div_acc = th.zeros(batch_size, device=device)
                    for i in range(n_acc):
                        z = (th.randn_like(x_noise) < 0).float() * 2.0 - 1.0
                        ut_dot_z = th.einsum(
                            "ij,ij->i",
                            ut.flatten(start_dim=1),
                            z.flatten(start_dim=1),
                        )
                        grad_ut_dot_z = gradient(
                            ut_dot_z,
                            xt_grad,
                            create_graph=False,
                            retain_graph=(i < n_acc - 1),
                        )
                        div_est = th.einsum(
                            "ij,ij->i",
                            grad_ut_dot_z.flatten(start_dim=1),
                            z.flatten(start_dim=1),
                        )
                        div_acc = div_acc + div_est
                    div = div_acc / n_acc

                ut = ut.detach()
                div = div.detach()

            return ut, div

        y_init = (x_noise, th.zeros(batch_size, device=device))
        t = self.t.to(device)

        sol, log_det = odeint(
            dynamics_with_divergence,
            y_init,
            t,
            method=self.sampler_type,
            atol=[self.atol, self.atol],
            rtol=[self.rtol, self.rtol],
        )

        log_likelihood = log_p_init + log_det[-1]

        return sol, log_likelihood
