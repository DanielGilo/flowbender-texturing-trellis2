import gc
from typing import *
import os
import copy
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from easydict import EasyDict as edict
import cv2

from trellis2.pipelines import samplers

from ...modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin, MultiImageConditionedMixin
from ...renderers import PbrMeshRenderer, EnvMap



class SparseFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            #num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            num_workers=min(8, int(np.ceil(os.cpu_count() / torch.cuda.device_count()))),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=functools.partial(self._get_dataset_attr('collate_fn'), split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def training_losses(
        self,
        x_0: sp.SparseTensor,
        ext_cond=None,
        t=None,
        noise=None,
        cond=None,           # kept to absorb the dataloader kwarg
        **kwargs
    ) -> Tuple[Dict, Dict]:

        x_t = self.diffuse(x_0, t, noise=noise)

        pred = self.training_models['denoiser'](x_t, t * 1000, ext_cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape

        terms = edict()
        target = self.get_v(x_0, noise, t)
        terms['mse'] = F.mse_loss(pred.feats, target.feats)

        # Guided iteration, or unguided (g_t zeroed in concat_cond by prep_for_training_iter).
        # Both use the standard FM loss.
        if self.mode == "vanillapp":
            # ControlNet++-style render consistency loss.
            # Surrogate gradient pattern: detach pred.feats as an isolated leaf so that
            # autograd.grad frees the entire decoder graph before l.backward() runs.
            # This avoids the sparse-op custom backward conflicting with the outer backward.
            # Mathematically exact: d(surrogate)/dθ = d(render_loss)/dθ by chain rule.
            t_per_point = torch.zeros(
                x_t.feats.shape[0], 1, device=x_t.feats.device, dtype=torch.float32
            )
            for i in range(x_0.shape[0]):
                t_per_point[x_0.layout[i]] = t[i].item()

            pred_feats_for_decode = pred.feats.detach().float().requires_grad_(True)
            pred_x0_feats = (
                (1.0 - self.sigma_min) * x_t.feats.detach().float()
                - (self.sigma_min + (1.0 - self.sigma_min) * t_per_point) * pred_feats_for_decode
            )
            pred_x0 = pred.replace(pred_x0_feats)
            shape_z = kwargs['concat_cond']

            torch.cuda.empty_cache()
            rendered = self.decode_and_render(pred_x0, shape_z, **kwargs)
            del pred_x0

            # Validity mask: skip failed renders and high-noise timesteps.
            render_ok = rendered.abs().sum(dim=(1, 2, 3)) > 0  # [B]
            t_ok = t <= self.render_loss_t_threshold  # [B]
            valid_mask = (render_ok & t_ok).float()[:, None, None, None]  # [B,1,1,1]

            # Foreground-only MSE: use cond to derive foreground mask (cond = shaded*alpha,
            # so foreground pixels are where cond > 0). Avoids background diluting the loss,
            # which would otherwise require an unnaturally large render_loss_weight.
            fg_mask = (cond.detach().abs().sum(dim=1, keepdim=True) > 1e-6).float()  # [B,1,H,W]
            pixel_mask = valid_mask * fg_mask  # [B,1,H,W]
            n_fg = pixel_mask.sum().clamp(min=1.0)
            render_loss = ((rendered - cond.detach()) ** 2 * pixel_mask).sum() / n_fg
            del rendered

            if render_loss.grad_fn is None:
                # decode_and_render returned a no-grad zeros tensor (decode_latent failed).
                print("[vanillapp] render_loss has no grad_fn (decode_latent failed). Skipping render term.")
                del pred_feats_for_decode
                terms['render_loss'] = torch.tensor(0.0, device=pred.feats.device)
                terms['loss'] = terms['mse']
            else:
                try:
                    torch.cuda.empty_cache()
                    # Constant factor purely for float32 numerical stability (avoids
                    # bfloat16 underflow in autograd.grad). render_loss_weight controls
                    # the actual surrogate importance below.
                    d_render = torch.autograd.grad(
                        100.0 * render_loss, pred_feats_for_decode, retain_graph=False
                    )[0].detach()
                except torch.OutOfMemoryError:
                    print("[vanillapp] OOM during autograd.grad backward. Skipping render term.")
                    del pred_feats_for_decode, render_loss
                    torch.cuda.empty_cache()
                    terms['render_loss'] = torch.tensor(0.0, device=pred.feats.device)
                    terms['loss'] = terms['mse']
                else:
                    del pred_feats_for_decode
                    torch.cuda.empty_cache()
                    # Grad norm of MSE term w.r.t. pred.feats: ||2*(pred-target)/N||
                    # Divide by 100 to undo the stability factor applied before autograd.grad.
                    mse_grad_norm = (2.0 * (pred.feats.detach().float() - target.feats.detach().float()) / pred.feats.numel()).norm().item()
                    terms['mse_grad_norm'] = torch.tensor(mse_grad_norm, device=pred.feats.device)
                    d_render_norm = d_render.float().norm().item()
                    if d_render_norm > 1e-8:
                        d_render = (d_render.float() / d_render_norm).to(pred.feats.dtype)
                    surrogate = (d_render.to(pred.feats.dtype) * pred.feats).sum()
                    del d_render
                    terms['render_loss'] = render_loss.detach()
                    terms['loss'] = terms['mse'] + self.render_loss_weight * surrogate
        else:
            terms['loss'] = terms['mse']

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f'bin_{i}'] = {'mse': mse_per_instance[time_bin == i].mean()}

        return terms, {}
    
    def prep_for_training_iter(self, x_0: sp.SparseTensor, cond, **kwargs):
        """
        Prepare for training iteration.
        """
        # 1. Setup noise and timesteps (to be reused in training step)
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()

        # 2. Diffuse and Predict
        x_t = self.diffuse(x_0, t, noise=noise)
        
        concat_cond = kwargs['concat_cond']
        
        if self.mode in ("vanilla", "vanillapp"):
            processed_cond = self.get_cond(cond) # encodes w dino + cfg logic
            return processed_cond, t, noise, concat_cond
        
        elif self.mode == "with_grad":
            with torch.no_grad():
                encoded_cond = self.encode_image(cond)
            p_uncond = getattr(self, 'p_uncond', 0.0)
            cond_processed = ClassifierFreeGuidanceMixin.get_cond(
                self, encoded_cond, neg_cond=torch.zeros_like(encoded_cond), p_uncond=p_uncond
            )

            p_unguided = getattr(self, 'p_unguided', 0.0)
            is_unguided = p_unguided > 0.0 and torch.rand(1).item() < p_unguided

            if is_unguided:
                shape_z = kwargs['concat_cond']
                zero_grad = self._make_zero_grad(shape_z)
                concat_cond = sp.sparse_cat([shape_z, zero_grad], dim=-1)
                return cond_processed, t, noise, concat_cond, {}

            loss_f = self.get_render_loss_fn()
            grad = self.get_loss_grad_wrt_x_t(
                x_t, t, encoded_cond=encoded_cond, cond=cond, loss_f=loss_f, **kwargs
            )
            concat_cond, extras = self._prepare_grad_concat_cond(
                grad, is_unguided=is_unguided, **kwargs
            )
            return cond_processed, t, noise, concat_cond, extras

        elif self.mode == "with_prox_grad":
            with torch.no_grad():
                encoded_cond = self.encode_image(cond)
            p_uncond = getattr(self, 'p_uncond', 0.0)
            cond_processed = ClassifierFreeGuidanceMixin.get_cond(
                self, encoded_cond, neg_cond=torch.zeros_like(encoded_cond), p_uncond=p_uncond
            )

            # Sample once here so both the early-return path and _prepare_grad_concat_cond
            # see the same guided/unguided decision.
            p_unguided = getattr(self, 'p_unguided', 0.0)
            is_unguided = p_unguided > 0.0 and torch.rand(1).item() < p_unguided

            if is_unguided:
                # Skip g_t computation — the training forward always uses g_t=0.
                shape_z = kwargs['concat_cond']
                zero_grad = self._make_zero_grad(shape_z)
                concat_cond = sp.sparse_cat([shape_z, zero_grad], dim=-1)
                return cond_processed, t, noise, concat_cond, {}

            # Guided: g_t is needed.
            loss_f = self.get_render_loss_fn()
            grad = self.get_loss_grad_wrt_pred_x_0(
                x_t, t, encoded_cond=encoded_cond, cond=cond, loss_f=loss_f, **kwargs
            )
            concat_cond, extras = self._prepare_grad_concat_cond(
                grad, is_unguided=is_unguided, **kwargs
            )
            return cond_processed, t, noise, concat_cond, extras

        raise ValueError(f"Unsupported mode: {self.mode}")
    
    def get_pred_x_0(self, x_t: sp.SparseTensor, t, encoded_cond, return_pred_v: bool = False, **kwargs):
        sampler = self.get_sampler()

        denoiser = self.models['denoiser']
        was_training = denoiser.training
        denoiser.eval()

        pass_kwargs = {k: v for k, v in kwargs.items()}
        if 'cond' in pass_kwargs:
            pass_kwargs.pop('cond')

        pred_v = denoiser(x_t, t * 1000, cond=encoded_cond, **pass_kwargs)
        t_broadcast = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        pred_x_0, _ = sampler._v_to_xstart_eps(x_t, t_broadcast, pred_v)

        shape_z = pass_kwargs['concat_cond'].cuda()

        latent_dim = x_t.feats.shape[-1]
        if shape_z.feats.shape[-1] > latent_dim:
            shape_z = shape_z.replace(shape_z.feats[:, :latent_dim])

        # FIX: Explicitly detach shape_z so the VAE decoder doesn't build a graph for it!
        shape_z = shape_z.replace(shape_z.feats.detach())

        denoiser.train(was_training)

        pred_v_out = pred_v.replace(pred_v.feats.detach()) if return_pred_v else None
        del pred_v

        return (pred_x_0, shape_z, pred_v_out) if return_pred_v else (pred_x_0, shape_z)

    def decode_and_render(self, z: sp.SparseTensor, shape_z: sp.SparseTensor, **kwargs):
        with torch.autocast(device_type='cuda', enabled=False):
            try:
                reps = self._get_dataset_attr('decode_latent')(z.float(), shape_z, batch_size=1)
            except (RuntimeError, torch.OutOfMemoryError) as e:
                print(f"Warning: decode_latent failed ('{type(e).__name__}: {e}'). Returning black images for all {z.shape[0]} samples.")
                torch.cuda.empty_cache()
                resolution = 512
                return torch.zeros((z.shape[0], 3, resolution, resolution), device=z.feats.device, dtype=torch.float32)

            cond_extrinsics = kwargs['cond_extrinsics'].cuda().float()
            cond_intrinsics = kwargs['cond_intrinsics'].cuda().float()
            
            # Check if conditional envmaps are provided in the batch
            use_dynamic_envmap = 'cond_envmap' in kwargs and kwargs['cond_envmap'] is not None
            if not use_dynamic_envmap:
                # Fallback to the static forest envmap if not provided
                if not hasattr(self, '_envmap'):
                    self._envmap = EnvMap(torch.tensor(
                        cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                        dtype=torch.float32, device='cuda'
                    ))
            
            local_renderer = PbrMeshRenderer()
            local_renderer.rendering_options.resolution = 512
            local_renderer.rendering_options.near = 1
            local_renderer.rendering_options.far = 100
            local_renderer.rendering_options.ssaa = 2
            local_renderer.rendering_options.peel_layers = 8
            
            images = []
            for i, representation in enumerate(reps):
                representation.vertices = representation.vertices.float()
                
                if not torch.isfinite(representation.vertices).all():
                    print(f"Warning: Non-finite vertices detected in sample {i}. Skipping render.")
                    resolution = local_renderer.rendering_options.resolution
                    images.append(torch.zeros((3, resolution, resolution), device=representation.vertices.device, dtype=torch.float32))
                    continue

                # Select the correct envmap for this item in the batch
                if use_dynamic_envmap:
                    # Create an EnvMap on the fly from the batched tensor
                    current_envmap = EnvMap(kwargs['cond_envmap'][i].cuda())
                else:
                    # Use the cached static envmap
                    current_envmap = self._envmap

                try:
                    res = local_renderer.render(representation, cond_extrinsics[i], cond_intrinsics[i], envmap=current_envmap)
                    shaded = res['shaded']
                    alpha = res['alpha'].clamp(0, 1)
                    
                    rendered_img = shaded * alpha
                    images.append(rendered_img)
                except (RuntimeError, torch.OutOfMemoryError) as e:
                    err_str = str(e).lower()
                    is_oom = (
                        isinstance(e, torch.OutOfMemoryError)
                        or "out of memory" in err_str
                        or "cudamalloc" in err_str          # nvdiffrast: "Cuda error: 2[cudaMalloc...]"
                        or "cuda error: 2" in err_str       # cudaErrorMemoryAllocation
                    )
                    is_mesh_error = "subtriangle count overflow" in err_str or "glerror" in err_str
                    if is_oom or is_mesh_error:
                        print(f"Warning: render error '{type(e).__name__}: {e}' encountered in sample {i}. Returning black image.")
                        resolution = local_renderer.rendering_options.resolution
                        images.append(torch.zeros((3, resolution, resolution), device=representation.vertices.device, dtype=torch.float32))
                        torch.cuda.empty_cache()
                    else:
                        raise e
                
            images = torch.stack(images, dim=0)

        # FIX: Aggressively delete local variables that hold massive graph pointers
        del local_renderer
        del reps
        
        return images

    def get_cond_matching_render(self, x_t: sp.SparseTensor, t, encoded_cond, return_pred_v: bool = False, **kwargs):
        result = self.get_pred_x_0(x_t, t, encoded_cond, return_pred_v=return_pred_v, **kwargs)
        if return_pred_v:
            pred_x_0, shape_z, pred_v_out = result
        else:
            pred_x_0, shape_z = result
            pred_v_out = None

        images = self.decode_and_render(pred_x_0.cuda(), shape_z, **kwargs)

        del pred_x_0, shape_z

        return (images, pred_v_out) if return_pred_v else images
    
    def get_loss_grad_wrt_x_t(self, x_t: sp.SparseTensor, t, encoded_cond, cond, loss_f, return_render=False, return_pred_v=False, **kwargs):
        # Freeze all model params so no weight-gradient graphs are built.
        # try/finally ensures they are always restored even on OOM.
        requires_grad_states = {}
        for model_name, model in self.models.items():
            for param_name, param in model.named_parameters():
                requires_grad_states[f"{model_name}.{param_name}"] = param.requires_grad
                param.requires_grad = False

        pred_v_unguided = None  # always bound even if try raises

        try:
            with torch.enable_grad():
                x_t_in = x_t.replace(x_t.feats.detach())
                x_t_in.feats.requires_grad_(True)

                zero_grad_sp = self._make_zero_grad(x_t_in)

                pass_kwargs = {k: v for k, v in kwargs.items()}
                pass_kwargs['concat_cond'] = sp.sparse_cat([pass_kwargs['concat_cond'], zero_grad_sp], dim=-1)

                if return_pred_v:
                    rendered_images, pred_v_unguided = self.get_cond_matching_render(
                        x_t_in, t, encoded_cond=encoded_cond, return_pred_v=True, **pass_kwargs)
                else:
                    rendered_images = self.get_cond_matching_render(x_t_in, t, encoded_cond=encoded_cond, **pass_kwargs)

                # Detect failed renders (black images) and zero their gradient contribution.
                render_ok = rendered_images.abs().sum(dim=(1, 2, 3)) > 0  # [B]
                if not render_ok.all():
                    n_bad = (~render_ok).sum().item()
                    print(f"[get_loss_grad_wrt_x_t] {n_bad}/{render_ok.shape[0]} renders are black. Zeroing gradient for those samples.")

                loss = loss_f(rendered_images, cond)
                _render_out = rendered_images.detach() if return_render else None
                del rendered_images

                if loss.grad_fn is None:
                    print("[get_loss_grad_wrt_x_t] All renders failed — loss has no grad_fn. Returning zero gradient.")
                    grad = torch.zeros_like(x_t_in.feats)
                else:
                    try:
                        grad_tuple = torch.autograd.grad(loss, x_t_in.feats, create_graph=False, retain_graph=False, allow_unused=True)
                        grad = grad_tuple[0]
                    except torch.OutOfMemoryError:
                        print("[get_loss_grad_wrt_x_t] OOM during autograd.grad. Returning zero gradient.")
                        torch.cuda.empty_cache()
                        grad = None

                    if grad is None:
                        grad = torch.zeros_like(x_t_in.feats)
                    else:
                        grad = torch.nan_to_num(grad)
                        if not render_ok.all():
                            for i in range(x_t_in.shape[0]):
                                if not render_ok[i]:
                                    grad[x_t_in.layout[i]] = 0.0

                del loss, x_t_in, zero_grad_sp, pass_kwargs

        finally:
            # Always restore requires_grad, even on OOM.
            for model_name, model in self.models.items():
                for param_name, param in model.named_parameters():
                    param.requires_grad = requires_grad_states[f"{model_name}.{param_name}"]

        grad_normalized = self._normalize_gradient(grad, x_t)
        output_tensor = x_t.replace(grad_normalized)
        del grad

        if return_render and return_pred_v:
            return output_tensor, _render_out, pred_v_unguided
        elif return_render:
            return output_tensor, _render_out
        elif return_pred_v:
            return output_tensor, pred_v_unguided
        else:
            return output_tensor

    def get_loss_grad_wrt_pred_x_0(self, x_t: sp.SparseTensor, t, encoded_cond, cond, loss_f, return_render=False, return_pred_v=False, _skip_grad_slot=False, **kwargs):
        pred_v_unguided = None  # always bound
        with torch.no_grad():
            x_t_in = x_t.replace(x_t.feats.detach())

            pass_kwargs = {k: v for k, v in kwargs.items()}
            if not _skip_grad_slot:
                # Gradient-conditioned modes have an expanded input layer; fill the
                # gradient slot with zeros for the first (unguided) pass.
                zero_grad_sp = self._make_zero_grad(x_t_in)
                pass_kwargs['concat_cond'] = sp.sparse_cat([pass_kwargs['concat_cond'], zero_grad_sp], dim=-1)

            result = self.get_pred_x_0(x_t_in, t, encoded_cond, return_pred_v=return_pred_v, **pass_kwargs)
            if return_pred_v:
                pred_x_0, shape_z, pred_v_unguided = result
            else:
                pred_x_0, shape_z = result
            
        with torch.enable_grad():
            z = pred_x_0.cuda()
            z_feats = z.feats.detach()
            z_feats.requires_grad_(True)
            z = z.replace(z_feats)

            rendered_images = self.decode_and_render(z, shape_z, **pass_kwargs)

            # decode_and_render returns black (zero) images for any sample whose render
            # failed (OOM, degenerate mesh). A black image produces a non-zero but
            # meaningless MSE gradient. Detect fully-black renders and zero them out
            # so those samples contribute zero gradient rather than corrupted signal.
            render_ok = rendered_images.abs().sum(dim=(1, 2, 3)) > 0  # [B]
            if not render_ok.all():
                n_bad = (~render_ok).sum().item()
                print(f"[get_loss_grad_wrt_pred_x_0] {n_bad}/{render_ok.shape[0]} renders are black (OOM/degenerate). Zeroing gradient for those samples.")

            loss = loss_f(rendered_images, cond)

            # If all renders failed, loss has no grad_fn (all inputs were detached zeros).
            # autograd.grad raises "does not require grad" when the *output* has no graph —
            # allow_unused=True only handles inputs not in the graph, not a detached output.
            if loss.grad_fn is None:
                print("[get_loss_grad_wrt_pred_x_0] All renders failed — loss has no grad_fn. Returning zero gradient.")
                grad = torch.zeros_like(z.feats)
            else:
                grad_tuple = torch.autograd.grad(loss, z.feats, create_graph=False, retain_graph=False, allow_unused=True)
                grad = grad_tuple[0]

                # None means z.feats wasn't in the computation graph (circuit breaker / disconnected render).
                if grad is None:
                    print("[get_loss_grad_wrt_pred_x_0] Gradient is None (disconnected graph). Returning zero gradient.")
                    grad = torch.zeros_like(z.feats)
                else:
                    grad = torch.nan_to_num(grad)
                    # Zero out gradients for samples whose render failed.
                    if not render_ok.all():
                        for i in range(z.shape[0]):
                            if not render_ok[i]:
                                grad[z.layout[i]] = 0.0
                
            _render_out = rendered_images.detach() if return_render else None
            del rendered_images
            del loss
            del z
            del shape_z
            del pass_kwargs
            del pred_x_0
            del x_t_in
            if not _skip_grad_slot:
                del zero_grad_sp

        grad_normalized = self._normalize_gradient(grad, x_t)
        output_tensor = x_t.replace(grad_normalized)
        del grad

        gc.collect()
        torch.cuda.empty_cache()

        if return_render and return_pred_v:
            return output_tensor, _render_out, pred_v_unguided
        elif return_render:
            return output_tensor, _render_out
        elif return_pred_v:
            return output_tensor, pred_v_unguided
        else:
            return output_tensor

    def get_loss_grad_from_cached_x0(self, cached_x0: sp.SparseTensor, cond, loss_f, **kwargs) -> sp.SparseTensor:
        """
        Computes the render-loss gradient w.r.t. a cached x_hat_0 (e.g. from the previous
        sampling step), without running the denoiser. Used by the prior-step shortcut
        sampler (see `sample_with_prior_step_shortcut`) to approximate the look-ahead
        gradient at t < t_thresh.
        """
        shape_z = kwargs['concat_cond']
        latent_dim = cached_x0.feats.shape[-1]
        if shape_z.feats.shape[-1] > latent_dim:
            shape_z = shape_z.replace(shape_z.feats[:, :latent_dim].detach())

        with torch.enable_grad():
            z_feats = cached_x0.feats.detach().float().requires_grad_(True)
            z = cached_x0.replace(z_feats)
            rendered = self.decode_and_render(z, shape_z, **kwargs)
            del z

            render_ok = rendered.abs().sum(dim=(1, 2, 3)) > 0
            render_loss = loss_f(rendered, cond)
            del rendered

            if render_loss.grad_fn is None:
                grad = torch.zeros_like(cached_x0.feats, dtype=torch.float32)
            else:
                try:
                    g = torch.autograd.grad(render_loss, z_feats, retain_graph=False, allow_unused=True)[0]
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    g = None
                if g is None:
                    grad = torch.zeros_like(cached_x0.feats, dtype=torch.float32)
                else:
                    grad = torch.nan_to_num(g)
                    if not render_ok.all():
                        for i in range(cached_x0.shape[0]):
                            if not render_ok[i]:
                                grad[cached_x0.layout[i]] = 0.0
            del render_loss, z_feats

        grad_normalized = self._normalize_gradient(grad.to(cached_x0.feats.dtype), cached_x0)
        output_tensor = cached_x0.replace(grad_normalized)
        del grad
        gc.collect()
        torch.cuda.empty_cache()
        return output_tensor

    @torch.no_grad()
    def sample_with_prior_step_shortcut(
        self, noise: sp.SparseTensor, cond, neg_cond, batch_data: dict,
        steps: int = 12, t_thresh: float = 0.0, guidance_strength: float = 1.0,
    ) -> sp.SparseTensor:
        """
        Euler sampling with the paper's prior-step feedback approximation ("Efficient
        Sampling via Prior-Step Feedback Approximation"): for t < t_thresh, the feedback
        gradient is derived from the previous step's cached x_hat_0 (`get_loss_grad_from_cached_x0`)
        instead of a fresh look-ahead pass, cutting such steps from 2 denoiser evaluations
        to 1.

        Note on `t` convention: this uses the codebase's convention (t=1 is pure noise at
        the start of sampling, t=0 is the clean sample at the end), not the paper's FM
        convention (t=0 to t=1), which runs in the opposite direction. `t_thresh=0.0`
        (default) always takes the full two-pass step (`get_loss_grad_wrt_x_t` /
        `get_loss_grad_wrt_pred_x_0`) — no step ever reaches `t < 0`. `t_thresh=1.0` uses
        the shortcut for every step after the first (which always runs the full pass, as
        there is no cached x_hat_0 yet). Not compatible with grad_cfg_strength != 1.0:
        there is no cached unguided velocity to blend with on shortcut steps.
        """
        if self.mode == 'with_grad':
            get_full_pass_grad = self.get_loss_grad_wrt_x_t
        elif self.mode == 'with_prox_grad':
            get_full_pass_grad = self.get_loss_grad_wrt_pred_x_0
        else:
            raise ValueError(f"Prior-step shortcut requires mode in ('with_grad', 'with_prox_grad'), got '{self.mode}'")

        sampler = self.get_sampler()
        denoiser = self.models['denoiser']
        loss_f = self.get_render_loss_fn()
        shape_z = batch_data['concat_cond']
        kwargs_no_cond = {k: v for k, v in batch_data.items() if k != 'cond'}

        x_t = noise
        prev_x0 = None
        t_seq = np.linspace(1.0, 0.0, steps + 1).tolist()
        for t_val, t_prev_val in ((t_seq[i], t_seq[i + 1]) for i in range(steps)):
            t_tensor = torch.full((x_t.shape[0],), t_val, device=x_t.device, dtype=torch.float32)
            if prev_x0 is None or t_val >= t_thresh:
                grad = get_full_pass_grad(x_t, t_tensor, encoded_cond=cond, loss_f=loss_f, **batch_data)
            else:
                grad = self.get_loss_grad_from_cached_x0(prev_x0, batch_data['cond'], loss_f, **kwargs_no_cond)

            step_kwargs = dict(kwargs_no_cond)
            step_kwargs['concat_cond'] = sp.sparse_cat([shape_z, grad], dim=1)
            out = sampler.sample_once(
                denoiser, x_t, t_val, t_prev_val, cond,
                neg_cond=neg_cond, guidance_strength=guidance_strength, grad_cfg_strength=1.0,
                **step_kwargs
            )
            x_t = out.pred_x_prev
            prev_x0 = out.pred_x_0.replace(out.pred_x_0.feats.detach())

        return x_t

    def get_render_loss_fn(self):
        """Returns the render-gradient loss function: MSE, summed over all pixels."""
        def mse_sum_loss(pred, target):
            return F.mse_loss(pred, target, reduction='sum')
        return mse_sum_loss
    
    def _normalize_gradient(self, grad: torch.Tensor, x_t: sp.SparseTensor) -> torch.Tensor:
        """
        Normalizes the gradient hint per sample: divide by per-sample std, no mean
        subtraction. Removes mesh-complexity scale while preserving exact gradient
        direction (signs and relative magnitudes always correct).
        """
        C = grad.shape[-1]
        output_grads = torch.zeros(
            grad.shape[0], C, dtype=grad.dtype, device=grad.device
        )

        for i in range(x_t.shape[0]):
            point_indices = x_t.layout[i]
            g = grad[point_indices]
            if g.numel() == 0:
                continue
            output_grads[point_indices] = g / (g.std() + 1e-8)

        return output_grads

    def _make_zero_grad(self, ref_sp: sp.SparseTensor) -> sp.SparseTensor:
        """Return a SparseTensor of latent_dim zero channels for the gradient hint slot."""
        latent_dim = ref_sp.feats.shape[-1]
        zero_feats = torch.zeros(
            ref_sp.feats.shape[0], latent_dim,
            dtype=ref_sp.feats.dtype, device=ref_sp.feats.device
        )
        return ref_sp.replace(zero_feats)

    def _prepare_grad_concat_cond(self, grad, is_unguided=False, **kwargs):
        """
        Shared post-gradient logic for with_grad and with_prox_grad modes.

        Applies the guided/unguided decision:
          - guided   (1 - p_unguided): use grad as-is, standard flow matching loss in training_losses.
          - unguided (p_unguided):    zero out grad, training_losses falls through to FM loss.

        is_unguided must be pre-sampled by the caller.
        """
        shape_z = kwargs['concat_cond']   # original shape latent (before grad is appended)

        if is_unguided:
            grad = grad.replace(torch.zeros_like(grad.feats))

        concat_cond = sp.sparse_cat([shape_z, grad], dim=1)
        return concat_cond, {}
    
    
    def get_sampler(self, **kwargs):
        return samplers.RenderAwareFlowEulerSampler(self.sigma_min)

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        # Fix seed for deterministic validation (e.g. view selection)
        rng_state = np.random.get_state()
        np.random.seed(self.rank * 1000 + 42)

        if self.val_dataset is not None:
            print("using val dataset")
            dataset = self.val_dataset
            shuffle = False
        else:
            print("using train dataset")
            dataset = copy.deepcopy(self.dataset)
            shuffle = False
            
        sampler = None
        if self.world_size > 1 and not shuffle:
            sampler = DistributedSampler(dataset, shuffle=False)
            
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and (sampler is None),
            sampler=sampler,
            num_workers=0,
            collate_fn=self._get_dataset_attr('collate_fn'),
        )
        data_iter = iter(dataloader)

        # inference
        sampler = self.get_sampler()
        sample = []
        cond_vis = []
        consistency_vis_list = [] # Accumulate consistency results
        all_batch_data = []
        
        for i in range(0, num_samples, batch_size):
            data = next(data_iter)
            current_batch = min(batch_size, num_samples - i)
            batch_data = {k: v[:current_batch] for k, v in data.items()}
            batch_data = recursive_to_device(batch_data, 'cuda')
            all_batch_data.append(batch_data.copy())
            noise = batch_data['x_0'].replace(torch.randn_like(batch_data['x_0'].feats))
            
            # --- New Validation: Check Renderer Consistency ---
            check_fn = self._get_dataset_attr('check_renderer_consistency')
            if check_fn is not None:
                # At the first step (0), do it for the entire set of validation images.
                # Otherwise, only check the first batch to save rendering time.
                if self.step == 0 or i == 0:
                    vis = check_fn(batch_data)
                    if vis is not None:
                        consistency_vis_list.append(vis)
            # --------------------------------------------------
            
            cond_vis.append(self.vis_cond(**batch_data))
            del batch_data['x_0']
            args = self.get_inference_cond(**batch_data) # encodes cond image and prepares neg_cond for cfg
            
            cond_update_fn = None
            if self.mode == 'with_grad':
                def update_fn(x_t, t, encoded_cond, current_neg_cond):
                    t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.float32)
                    loss_f = self.get_render_loss_fn()
                    grad = self.get_loss_grad_wrt_x_t(x_t, t_tensor, encoded_cond=encoded_cond, loss_f=loss_f, **batch_data)
                    concat_cond = sp.sparse_cat([batch_data['concat_cond'], grad], dim=1)
                    return encoded_cond, current_neg_cond, concat_cond
                cond_update_fn = update_fn
                
            elif self.mode == 'with_prox_grad':
                def update_fn(x_t, t, encoded_cond, current_neg_cond):
                    t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.float32)
                    loss_f = self.get_render_loss_fn()
                    grad = self.get_loss_grad_wrt_pred_x_0(x_t, t_tensor, encoded_cond=encoded_cond, loss_f=loss_f, **batch_data)
                    concat_cond = sp.sparse_cat([batch_data['concat_cond'], grad], dim=1)
                    return encoded_cond, current_neg_cond, concat_cond
                cond_update_fn = update_fn

            self.models['denoiser'].eval()

            guidance_strength = 1.0
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                cond_update_fn=cond_update_fn,
                steps=12, guidance_strength=guidance_strength, verbose=verbose,
            )
            self.models['denoiser'].train()
            sample.append(res.samples)
            del res

            # Explicit cleanup to prevent memory accumulation
            del batch_data
            del args
            del noise
            del cond_update_fn
            torch.cuda.empty_cache()
            
        sample = sp.sparse_cat(sample)

        # Aggregate all batch data to match the full sample size
        full_data = {}
        if all_batch_data:
            for k in all_batch_data[0].keys():
                vals = [d[k] for d in all_batch_data]
                if isinstance(vals[0], sp.SparseTensor):
                    full_data[k] = sp.sparse_cat(vals, dim=0)
                elif isinstance(vals[0], torch.Tensor):
                    full_data[k] = torch.cat(vals, dim=0)
                else:
                    full_data[k] = sum(vals, [])

        sample_gt = full_data.copy()
        sample = {k: v if k != 'x_0' else sample for k, v in full_data.items()}
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        if consistency_vis_list:
            sample_dict['renderer_consistency'] = {'value': torch.cat(consistency_vis_list, dim=0), 'type': 'image'}
        
        # Restore random state
        np.random.set_state(rng_state)

        return sample_dict

class SparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SparseFlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    def get_sampler(self, **kwargs):
        return samplers.RenderAwareFlowEulerSampler(self.sigma_min)


class TextConditionedSparseFlowMatchingCFGTrainer(TextConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        text_cond_model(str): Text conditioning model.
    """
    pass


class ImageConditionedSparseFlowMatchingCFGTrainer(ImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with flow matching objective and classifier-free guidance.

    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass


class MultiImageConditionedSparseFlowMatchingCFGTrainer(MultiImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass
