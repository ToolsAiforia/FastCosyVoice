# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F
from matcha.models.components.flow_matching import BASECFM
from cosyvoice.utils.common import set_all_random_seed


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, cache=torch.zeros(1, 80, 0, 2)):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        cache_size = cache.shape[2]
        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        # NOTE when flow run in amp mode, x.dtype is float32, which cause nan in trt fp16 inference, so set dtype=spks.dtype
        # CFG packs [conditioned | unconditioned] along batch dim. For B requests → 2*B rows.
        B = x.size(0)
        x_in = torch.zeros([2 * B, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        mask_in = torch.zeros([2 * B, 1, x.size(2)], device=x.device, dtype=spks.dtype)
        mu_in = torch.zeros([2 * B, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        t_in = torch.zeros([2 * B], device=x.device, dtype=spks.dtype)
        spks_in = torch.zeros([2 * B, 80], device=x.device, dtype=spks.dtype)
        cond_in = torch.zeros([2 * B, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox.
            # Rows [0:B] = conditioned, [B:2B] = unconditioned (zeros for mu/spks/cond).
            # For B=1 this matches the original [2,...] shape behavior.
            x_in[:B] = x
            x_in[B:] = x
            mask_in[:B] = mask
            mask_in[B:] = mask
            mu_in[:B] = mu        # mu_in[B:] stays zeros (unconditioned)
            t_in[:] = t.unsqueeze(0)
            spks_in[:B] = spks    # spks_in[B:] stays zeros
            cond_in[:B] = cond    # cond_in[B:] stays zeros
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in,
                streaming
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()

    def forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        else:
            # Path D micro-optimization (2026-05-20): execute TRT on current_stream
            # without per-call synchronize(). Original code did 2 syncs per call
            # × 10 solve_euler iterations × N tokens = huge Python overhead.
            # TRT execute_async_v3 enqueues on current_stream; subsequent torch
            # ops on the same stream are correctly ordered without explicit sync.
            [estimator, stream], trt_engine = self.estimator.acquire_estimator()
            two_B = x.size(0)
            estimator.set_input_shape('x', (two_B, 80, x.size(2)))
            estimator.set_input_shape('mask', (two_B, 1, x.size(2)))
            estimator.set_input_shape('mu', (two_B, 80, x.size(2)))
            estimator.set_input_shape('t', (two_B,))
            estimator.set_input_shape('spks', (two_B, 80))
            estimator.set_input_shape('cond', (two_B, 80, x.size(2)))
            # Skip redundant .contiguous() — inputs from solve_euler pre-allocated buffers
            # are already contiguous; check first to avoid unnecessary copies.
            x_c = x if x.is_contiguous() else x.contiguous()
            mask_c = mask if mask.is_contiguous() else mask.contiguous()
            mu_c = mu if mu.is_contiguous() else mu.contiguous()
            t_c = t if t.is_contiguous() else t.contiguous()
            spks_c = spks if spks.is_contiguous() else spks.contiguous()
            cond_c = cond if cond.is_contiguous() else cond.contiguous()
            out = torch.empty_like(x_c)

            estimator.set_tensor_address('x', x_c.data_ptr())
            estimator.set_tensor_address('mask', mask_c.data_ptr())
            estimator.set_tensor_address('mu', mu_c.data_ptr())
            estimator.set_tensor_address('t', t_c.data_ptr())
            estimator.set_tensor_address('spks', spks_c.data_ptr())
            estimator.set_tensor_address('cond', cond_c.data_ptr())
            estimator.set_tensor_address('estimator_out', out.data_ptr())
            assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
            # No sync — out is queued on current_stream, downstream ops read it correctly.
            self.estimator.release_estimator(estimator, stream)
            return out

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, streaming=False):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        set_all_random_seed(0)
        self.rand_noise = torch.randn([1, 80, 50 * 300])

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, streaming=False):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        # Same fixed noise pattern for all rows in a batch (deterministic per-stream).
        z = self.rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype) * temperature
        if mu.size(0) > 1:
            z = z.expand(mu.size(0), -1, -1).contiguous()
        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, streaming=streaming), None
