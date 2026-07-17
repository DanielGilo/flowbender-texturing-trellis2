from abc import abstractmethod
import os
import time
import json
import copy
import threading
from functools import partial
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from torchvision import utils
from torch.utils.tensorboard import SummaryWriter

from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler
from ..utils.dist_utils import *
from ..utils import grad_clip_utils, elastic_utils
from ..modules import sparse as sp


class BasicTrainer:
    """
    Trainer for basic training loop.
    
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
        mix_precision_mode (str):
            - None: No mixed precision.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        mix_precision_dtype (str): Mixed precision dtype.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        parallel_mode (str): Parallel mode. Options are 'ddp'.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
    """
    def __init__(self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode=None,
        mix_precision_mode='inflat_all',
        mix_precision_dtype='float16',
        fp16_scale_growth=1e-3,
        parallel_mode='ddp',
        finetune_ckpt=None,
        log_param_stats=False,
        prefetch_data=True,
        snapshot_batch_size=4,
        i_print=1000,
        val_samples=None,
        max_samples=None,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
        mode="vanilla",
        render_loss_weight=1.0,
        render_loss_t_threshold=1.0,
        wandb_cfg=None,
        **kwargs
    ):
        assert batch_size is not None or batch_size_per_gpu is not None, 'Either batch_size or batch_size_per_gpu must be specified.'

        self.models = models
        self.dataset = dataset
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        if fp16_mode is not None:
            mix_precision_dtype = 'float16'
            mix_precision_mode = fp16_mode
        self.mix_precision_mode = mix_precision_mode
        self.mix_precision_dtype = str_to_dtype(mix_precision_dtype)
        self.fp16_scale_growth = fp16_scale_growth
        self.parallel_mode = parallel_mode
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        self.snapshot_batch_size = snapshot_batch_size
        self.val_samples = val_samples
        self.max_samples = max_samples
        self.log = []
        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck   
        self.mode = mode
        self.render_loss_weight = render_loss_weight
        self.render_loss_t_threshold = render_loss_t_threshold
        self.wandb_cfg = wandb_cfg

        if dist.is_initialized():
            # Multi-GPU params
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            # Single-GPU params
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        assert self.batch_size % self.world_size == 0, 'Batch size must be divisible by the number of GPUs.'
        assert self.batch_size_per_gpu % self.batch_split == 0, 'Batch size per GPU must be divisible by batch split.'

        # 1. Load base weights if finetuning (before LoRA injection and Optimizer init)
        if finetune_ckpt is not None:
            self._load_finetune_weights(finetune_ckpt)

        # 1.5 Expand input dimension if mode requires gradient conditioning
        if self.mode in ['with_grad', 'with_prox_grad']:
            self._expand_input_dimension()

        # 2. Inject LoRA if applicable
        for model in self.models.values():
            if hasattr(model, 'inject_lora'):
                model.inject_lora()

        # 2.5 Unfreeze input layer for gradient-conditioned modes
        if self.mode in ['with_grad', 'with_prox_grad'] and 'denoiser' in self.models:
            denoiser = self.models['denoiser']
            if hasattr(denoiser, 'input_layer'):
                layer = denoiser.input_layer
                # If LoRA wrapped it, unfreeze base and disable adapters to train full rank
                if hasattr(layer, 'base_layer'):
                    layer.base_layer.weight.requires_grad = True
                    if hasattr(layer.base_layer, 'bias') and layer.base_layer.bias is not None:
                        layer.base_layer.bias.requires_grad = True
                    # Revert to base layer to bypass LoRA entirely
                    denoiser.input_layer = layer.base_layer
                else:
                    layer.requires_grad_(True)

        # Limit training samples if max_samples is set
        if self.max_samples is not None and self.max_samples > 0:
            if self.is_master:
                print(f"Limiting training dataset to the first {self.max_samples} samples.")
            self.dataset = torch.utils.data.Subset(self.dataset, range(min(self.max_samples, len(self.dataset))))

        # Split dataset if val_samples is set
        self.val_dataset = None
        if self.val_samples is not None and self.val_samples > 0:
            indices = np.arange(len(self.dataset))
            val_indices = indices[:self.val_samples]
            train_indices = indices[self.val_samples:]
            self.train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
            self.val_dataset = torch.utils.data.Subset(self.dataset, val_indices)
            self.dataset = self.train_dataset

        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)
        
        # Load checkpoint
        self.step = 0
        if load_dir is not None and step is not None:
            self.load(load_dir, step)
        
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
            if self.wandb_cfg is not None:
                import wandb
                wandb.init(dir=self.output_dir, **self.wandb_cfg)
            else:
                self.writer = SummaryWriter(os.path.join(self.output_dir, 'tb_logs'))

        if self.parallel_mode == 'ddp' and self.world_size > 1:
            self.check_ddp()
            
        if self.is_master:
            print('\n\nTrainer initialized.')
            print(self)

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Models:')
        for name, model in self.models.items():
            lines.append(f'    - {name}: {model.__class__.__name__}')
        lines.append(f'  - Dataset: {indent(str(self.dataset), 2)}')
        lines.append(f'  - Dataloader:')
        lines.append(f'    - Sampler: {self.dataloader.sampler.__class__.__name__}')
        lines.append(f'    - Num workers: {self.dataloader.num_workers}')
        lines.append(f'  - Number of steps: {self.max_steps}')
        lines.append(f'  - Number of GPUs: {self.world_size}')
        lines.append(f'  - Batch size: {self.batch_size}')
        lines.append(f'  - Batch size per GPU: {self.batch_size_per_gpu}')
        lines.append(f'  - Batch split: {self.batch_split}')
        lines.append(f'  - Optimizer: {self.optimizer.__class__.__name__}')
        lines.append(f'  - Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        if self.lr_scheduler_config is not None:
            lines.append(f'  - LR scheduler: {self.lr_scheduler.__class__.__name__}')
        if self.elastic_controller_config is not None:
            lines.append(f'  - Elastic memory: {indent(str(self.elastic_controller), 2)}')
        if self.grad_clip is not None:
            lines.append(f'  - Gradient clip: {indent(str(self.grad_clip), 2)}')
        lines.append(f'  - EMA rate: {self.ema_rate}')
        lines.append(f'  - Mixed precision dtype: {self.mix_precision_dtype}')
        lines.append(f'  - Mixed precision mode: {self.mix_precision_mode}')
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            lines.append(f'  - FP16 scale growth: {self.fp16_scale_growth}')
        lines.append(f'  - Parallel mode: {self.parallel_mode}')
        lines.append(f'  - p_uncond: {getattr(self, "p_uncond", 0.0)}')
        lines.append(f'  - p_unguided: {getattr(self, "p_unguided", 0.0)}')
        return '\n'.join(lines)

    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, 'device'):
                return model.device
        return next(list(self.models.values())[0].parameters()).device
            
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        if self.world_size > 1:
            # Prepare distributed data parallel
            self.training_models = {
                name: DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    bucket_cap_mb=128,
                    find_unused_parameters=False
                )
                for name, model in self.models.items()
            }
        else:
            self.training_models = self.models

        # Build master params
        self.model_params = sum(
            [[p for p in model.parameters() if p.requires_grad] for model in self.models.values()]
        , [])
        if self.mix_precision_mode == 'amp':
            self.master_params = self.model_params
            if self.mix_precision_dtype == torch.float16:
                self.scaler = torch.GradScaler()
        elif self.mix_precision_mode == 'inflat_all':
            self.master_params = make_master_params(self.model_params)
            if self.mix_precision_dtype == torch.float16:
                self.log_scale = 20.0
        elif self.mix_precision_mode is None:
            self.master_params = self.model_params
        else:
            raise NotImplementedError(f'Mix precision mode {self.mix_precision_mode} is not implemented.')

        # Build EMA params
        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]

        # Initialize optimizer
        if hasattr(torch.optim, self.optimizer_config['name']):
            self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(self.master_params, **self.optimizer_config['args'])
        else:
            self.optimizer = globals()[self.optimizer_config['name']](self.master_params, **self.optimizer_config['args'])
        
        # Initalize learning rate scheduler
        if self.lr_scheduler_config is not None:
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name']):
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])(self.optimizer, **self.lr_scheduler_config['args'])
            else:
                self.lr_scheduler = globals()[self.lr_scheduler_config['name']](self.optimizer, **self.lr_scheduler_config['args'])

        # Initialize elastic memory controller
        if self.elastic_controller_config is not None:
            assert any([isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)) for model in self.models.values()]), \
                'No elastic module found in models, please inherit from ElasticModule or ElasticModuleMixin'
            self.elastic_controller = getattr(elastic_utils, self.elastic_controller_config['name'])(**self.elastic_controller_config['args'])
            for model in self.models.values():
                if isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)):
                    model.register_memory_controller(self.elastic_controller)

        # Initialize gradient clipper
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, (float, int)):
                self.grad_clip = float(self.grad_clip)
            else:
                self.grad_clip = getattr(grad_clip_utils, self.grad_clip['name'])(**self.grad_clip['args'])

    def _expand_input_dimension(self):
        if 'denoiser' not in self.models:
            return

        model = self.models['denoiser']
        if hasattr(model, 'input_layer') and isinstance(model.input_layer, sp.SparseLinear):
            old_layer = model.input_layer

            # Original: [x, concat_cond] → 2 * latent_dim
            # Expanded: [x, concat_cond, grad_hint] → 3 * latent_dim
            latent_dim = old_layer.in_features // 2
            new_in_features = old_layer.in_features + latent_dim

            if self.is_master:
                print(f"Expanding input layer from {old_layer.in_features} to {new_in_features} channels for {self.mode} mode.")
            
            new_layer = sp.SparseLinear(new_in_features, old_layer.out_features, bias=old_layer.bias is not None)
            new_layer.to(old_layer.weight.device)
            
            with torch.no_grad():
                new_layer.weight[:, :old_layer.in_features] = old_layer.weight
                nn.init.zeros_(new_layer.weight[:, old_layer.in_features:])
                if old_layer.bias is not None:
                    new_layer.bias.copy_(old_layer.bias)
            
            model.input_layer = new_layer
            if hasattr(model, 'in_channels'):
                model.in_channels = new_in_features

    def _get_dataset_attr(self, attr):
        if hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        if hasattr(self.dataset, 'dataset') and hasattr(self.dataset.dataset, attr):
            return getattr(self.dataset.dataset, attr)
        return None

    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = ResumableSampler(
            self.dataset,
            shuffle=True,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            #num_workers=int(np.ceil(os.cpu_count() / torch.cuda.device_count())),
            num_workers=min(8, int(np.ceil(os.cpu_count() / torch.cuda.device_count()))),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=self._get_dataset_attr('collate_fn'),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def _master_params_to_state_dicts(self, master_params):
        """
        Convert master params to dict of state_dicts.
        """
        if self.mix_precision_mode == 'inflat_all':
            master_params = unflatten_master_params(self.model_params, master_params)
        state_dicts = {name: model.state_dict() for name, model in self.models.items()}
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        for i, (model_name, param_name) in enumerate(master_params_names):
            state_dicts[model_name][param_name] = master_params[i]
        return state_dicts

    def _state_dicts_to_master_params(self, master_params, state_dicts):
        """
        Convert a state_dict to master params.
        """
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        params = [state_dicts[name][param_name] for name, param_name in master_params_names]
        if self.mix_precision_mode == 'inflat_all':
            model_params_to_master_params(params, master_params)
        else:
            for i, param in enumerate(params):
                master_params[i].data.copy_(param.data)

    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print(f'\nLoading checkpoint from step {step}...', end='')
            
        model_ckpts = {}
        for name, model in self.models.items():
            model_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'{name}_step{step:07d}.pt')), map_location=self.device, weights_only=True)
            model_ckpts[name] = model_ckpt
            # strict=False allows loading checkpoints that only contain LoRA adapters (The Progress)
            model.load_state_dict(model_ckpt, strict=False)
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = {}
                for name, model in self.models.items():
                    ema_ckpt = torch.load(os.path.join(load_dir, 'ckpts', f'{name}_ema{ema_rate}_step{step:07d}.pt'), map_location=self.device, weights_only=True)
                    ema_ckpts[name] = ema_ckpt
                self._state_dicts_to_master_params(self.ema_params[i], ema_ckpts)
                del ema_ckpts
        
        misc_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')), map_location=torch.device('cpu'), weights_only=False)
        self.optimizer.load_state_dict(misc_ckpt['optimizer'])
        self.step = misc_ckpt['step']
        self.data_sampler.load_state_dict(misc_ckpt['data_sampler'])
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            self.scaler.load_state_dict(misc_ckpt['scaler'])
        elif self.mix_precision_mode == 'inflat_all' and self.mix_precision_dtype == torch.float16:
            self.log_scale = misc_ckpt['log_scale']
        if self.lr_scheduler_config is not None:
            self.lr_scheduler.load_state_dict(misc_ckpt['lr_scheduler'])
        if self.elastic_controller_config is not None:
            self.elastic_controller.load_state_dict(misc_ckpt['elastic_controller'])
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            self.grad_clip.load_state_dict(misc_ckpt['grad_clip'])
        del misc_ckpt

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(' Done.')

        if self.world_size > 1:
            self.check_ddp()

    def save(self, non_blocking=True):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        assert self.is_master, 'save() should be called only by the rank 0 process.'
        print(f'\nSaving checkpoint at step {self.step}...', end='')
        
        model_ckpts = self._master_params_to_state_dicts(self.master_params)
        for name, model_ckpt in model_ckpts.items():
            model_ckpt = {k: v.cpu() for k, v in model_ckpt.items()}  # Move to CPU for saving
            if non_blocking:
                threading.Thread(
                    target=torch.save,
                    args=(model_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_step{self.step:07d}.pt')),
                ).start()
            else:
                torch.save(model_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_step{self.step:07d}.pt'))
        
        for i, ema_rate in enumerate(self.ema_rate):
            ema_ckpts = self._master_params_to_state_dicts(self.ema_params[i])
            for name, ema_ckpt in ema_ckpts.items():
                ema_ckpt = {k: v.cpu() for k, v in ema_ckpt.items()}  # Move to CPU for saving
                if non_blocking:
                    threading.Thread(
                        target=torch.save,
                        args=(ema_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_ema{ema_rate}_step{self.step:07d}.pt')),
                    ).start()
                else:
                    torch.save(ema_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_ema{ema_rate}_step{self.step:07d}.pt'))

        misc_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'data_sampler': self.data_sampler.state_dict(),
        }
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            misc_ckpt['scaler'] = self.scaler.state_dict()
        elif self.mix_precision_mode == 'inflat_all' and self.mix_precision_dtype == torch.float16:
            misc_ckpt['log_scale'] = self.log_scale
        if self.lr_scheduler_config is not None:
            misc_ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()
        if self.elastic_controller_config is not None:
            misc_ckpt['elastic_controller'] = self.elastic_controller.state_dict()
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            misc_ckpt['grad_clip'] = self.grad_clip.state_dict()
        if non_blocking:
            threading.Thread(
                target=torch.save,
                args=(misc_ckpt, os.path.join(self.output_dir, 'ckpts', f'misc_step{self.step:07d}.pt')),
            ).start()
        else:
            torch.save(misc_ckpt, os.path.join(self.output_dir, 'ckpts', f'misc_step{self.step:07d}.pt'))
        print(' Done.')

    def _load_checkpoint(self, path):
        if os.path.exists(path):
            if path.endswith('.safetensors'):
                from safetensors.torch import load_file
                return load_file(path, device=str(self.device))
            else:
                return torch.load(read_file_dist(path), map_location=self.device, weights_only=True)
        
        # Try loading from huggingface
        if len(path.split('/')) > 1:
            try:
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file
                
                parts = path.split('/')
                repo_id = '/'.join(parts[:2])
                filename = '/'.join(parts[2:])
                
                if filename.endswith('.safetensors'):
                    cached_file = hf_hub_download(repo_id, filename)
                    return load_file(cached_file, device=str(self.device))
                elif filename.endswith('.pt') or filename.endswith('.pth') or filename.endswith('.bin'):
                    cached_file = hf_hub_download(repo_id, filename)
                    return torch.load(cached_file, map_location=self.device, weights_only=True)
                else:
                    cached_file = hf_hub_download(repo_id, f"{filename}.safetensors")
                    return load_file(cached_file, device=str(self.device))
            except Exception:
                pass

        return torch.load(read_file_dist(path), map_location=self.device, weights_only=True)

    def _load_finetune_weights(self, finetune_ckpt):
        """
        Load weights from finetune_ckpt into models.
        """
        if self.is_master:
            print('\nFinetuning from:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')
        
        for name, model in self.models.items():
            if name in finetune_ckpt:
                model_ckpt = self._load_checkpoint(finetune_ckpt[name])
                model_state_dict = model.state_dict()
                for k, v in model_ckpt.items():
                    if k not in model_state_dict:
                        if self.is_master:
                            print(f'Warning: {k} not found in model_state_dict, skipped.')
                        model_ckpt[k] = None
                    elif model_ckpt[k].shape != model_state_dict[k].shape:
                        if self.is_master:
                            print(f'Warning: {k} shape mismatch, {model_ckpt[k].shape} vs {model_state_dict[k].shape}, skipped.')
                        model_ckpt[k] = model_state_dict[k]
                model_ckpt = {k: v for k, v in model_ckpt.items() if v is not None}
                model.load_state_dict(model_ckpt, strict=False)
            else:
                if self.is_master:
                    print(f'Warning: {name} not found in finetune_ckpt, skipped.')
        
        if self.world_size > 1:
            dist.barrier()

    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print('\nFinetuning from:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')
        
        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name in finetune_ckpt:
                model_ckpt = self._load_checkpoint(finetune_ckpt[name])
                for k, v in model_ckpt.items():
                    if k not in model_state_dict:
                        if self.is_master:
                            print(f'Warning: {k} not found in model_state_dict, skipped.')
                        model_ckpt[k] = None
                    elif model_ckpt[k].shape != model_state_dict[k].shape:
                        if self.is_master:
                            print(f'Warning: {k} shape mismatch, {model_ckpt[k].shape} vs {model_state_dict[k].shape}, skipped.')
                        model_ckpt[k] = model_state_dict[k]
                model_ckpt = {k: v for k, v in model_ckpt.items() if v is not None}
                model_ckpts[name] = model_ckpt
                model.load_state_dict(model_ckpt)
            else:
                if self.is_master:
                    print(f'Warning: {name} not found in finetune_ckpt, skipped.')
                model_ckpts[name] = model_state_dict
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                self._state_dicts_to_master_params(self.ema_params[i], model_ckpts)
        del model_ckpts

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print('Done.')

        if self.world_size > 1:
            self.check_ddp()

    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):
        """
        Run a snapshot of the model.
        """
        pass

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image.
        """
        visualize_sample_fn = self._get_dataset_attr('visualize_sample')
        if visualize_sample_fn is not None:
            return visualize_sample_fn(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100, batch_size=4):
        """
        Sample images from the dataset.
        """
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=1,
            shuffle=True,
            collate_fn=self._get_dataset_attr('collate_fn'),
        )
        save_cfg = {}
        for i in range(0, num_samples, batch_size):
            data = next(iter(dataloader))
            data = {k: v[:min(num_samples - i, batch_size)] for k, v in data.items()}
            data = recursive_to_device(data, self.device)
            vis = self.visualize_sample(data)
            if isinstance(vis, dict):
                for k, v in vis.items():
                    if f'dataset_{k}' not in save_cfg:
                        save_cfg[f'dataset_{k}'] = []
                    save_cfg[f'dataset_{k}'].append(v)
            else:
                if 'dataset' not in save_cfg:
                    save_cfg['dataset'] = []
                save_cfg['dataset'].append(vis)
        for name, image in save_cfg.items():
            utils.save_image(
                torch.cat(image, dim=0),
                os.path.join(self.output_dir, 'samples', f'{name}.jpg'),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self._get_dataset_attr('value_range'),
            )

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=64, batch_size=4, verbose=False):
        """
        Sample images from the model.
        NOTE: This function should be called by all processes.
        """
        if self.is_master:
            print(f'\nSampling {num_samples} images...', end='')

        if suffix is None:
            suffix = f'step{self.step:07d}'

        # Assign tasks
        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        amp_context = partial(torch.autocast, device_type='cuda', dtype=self.mix_precision_dtype) if self.mix_precision_mode == 'amp' else nullcontext
        with amp_context():
            samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose)

        # Preprocess images
        for key in list(samples.keys()):
            if samples[key]['type'] == 'sample':
                sample_val = samples[key]['value']
                vis = self.visualize_sample(sample_val)
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                    del samples[key]
                else:
                    samples[key] = {'value': vis, 'type': 'image'}
                
                visualize_cond_sample_fn = self._get_dataset_attr('visualize_cond_sample')
                if visualize_cond_sample_fn is not None:
                    vis_cond = visualize_cond_sample_fn(sample_val) #TODO: repeats decoding, fix (unify with visualize_sample probably)
                    if isinstance(vis_cond, dict):
                        for k, v in vis_cond.items():
                            samples[f'{key}_cond_{k}'] = {'value': v, 'type': 'image'}
                    else:
                        samples[f'{key}_cond'] = {'value': vis_cond, 'type': 'image'}

        # Gather results
        if self.world_size > 1:
            for key in samples.keys():
                samples[key]['value'] = samples[key]['value'].contiguous()
                if self.is_master:
                    all_images = [torch.empty_like(samples[key]['value']) for _ in range(self.world_size)]
                else:
                    all_images = []
                dist.gather(samples[key]['value'], all_images, dst=0)
                if self.is_master:
                    samples[key]['value'] = torch.cat(all_images, dim=0)[:num_samples]

        # Save images
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'samples', suffix), exist_ok=True)
            wandb_log_data = {}
            
            # Log alignment PSNR
            if 'image' in samples and 'sample_cond_shaded' in samples:
                cond = samples['image']['value'].float()
                render = samples['sample_cond_shaded']['value'].float()
                if cond.shape != render.shape:
                    print(f'Warning: cond and render shape mismatch, {cond.shape} vs {render.shape}, resizing cond with bilinear interpolation.')
                    cond = F.interpolate(cond, size=render.shape[-2:], mode='bilinear', align_corners=False)
                mse = torch.mean((cond - render) ** 2, dim=(1, 2, 3))
                psnr = torch.mean(-10 * torch.log10(mse + 1e-8))
                
                # Calculate LPIPS
                if not hasattr(self, 'val_lpips_loss_fn'):
                    import lpips
                    self.val_lpips_loss_fn = lpips.LPIPS(net='vgg').to(cond.device)
                    self.val_lpips_loss_fn.eval()
                    self.val_lpips_loss_fn.requires_grad_(False)
                lpips_val = self.val_lpips_loss_fn(render * 2.0 - 1.0, cond * 2.0 - 1.0).mean().item()
                
                # Calculate CLIP
                # if not hasattr(self, 'val_clip_model'):
                #     from transformers import CLIPVisionModelWithProjection
                #     import torchvision.transforms as T
                #     self.val_clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(cond.device, dtype=torch.float32)
                #     self.val_clip_model.eval()
                #     self.val_clip_model.requires_grad_(False)
                #     self.val_clip_normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                #                                           std=[0.26862954, 0.26130258, 0.27577711])
                # render_proc = self.val_clip_normalize(F.interpolate(render, size=(224, 224), mode='bilinear', align_corners=False))
                # cond_proc = self.val_clip_normalize(F.interpolate(cond, size=(224, 224), mode='bilinear', align_corners=False))
                # render_feat = self.val_clip_model(render_proc).image_embeds
                # cond_feat = self.val_clip_model(cond_proc).image_embeds
                # clip_val = (1.0 - F.cosine_similarity(render_feat, cond_feat, dim=-1)).mean().item()
                
                if self.wandb_cfg is not None:
                    wandb_log_data['val/psnr_alignment'] = psnr
                    wandb_log_data['val/lpips_alignment'] = lpips_val
                    #wandb_log_data['val/clip_alignment'] = clip_val
                else:
                    self.writer.add_scalar('val/psnr_alignment', psnr, self.step)
                    self.writer.add_scalar('val/lpips_alignment', lpips_val, self.step)
                    #self.writer.add_scalar('val/clip_alignment', clip_val, self.step)

            # Log base color PSNR
            if 'sample_base_color' in samples and 'sample_gt_base_color' in samples:
                pred_bc = samples['sample_base_color']['value'].float()
                gt_bc = samples['sample_gt_base_color']['value'].float()
                if pred_bc.shape != gt_bc.shape:
                    print(f'Warning: pred_bc and gt_bc shape mismatch, {pred_bc.shape} vs {gt_bc.shape}, resizing pred_bc with bilinear interpolation.')
                    pred_bc = F.interpolate(pred_bc, size=gt_bc.shape[-2:], mode='bilinear', align_corners=False)
                mse_bc = torch.mean((pred_bc - gt_bc) ** 2, dim=(1, 2, 3))
                psnr_bc = torch.mean(-10 * torch.log10(mse_bc + 1e-8))
                if self.wandb_cfg is not None:
                    wandb_log_data['val/psnr_base_color'] = psnr_bc
                else:
                    self.writer.add_scalar('val/psnr_base_color', psnr_bc, self.step)

            for key in samples.keys():
                if samples[key]['type'] == 'image':
                    grid = utils.make_grid(
                        samples[key]['value'],
                        nrow=int(np.sqrt(num_samples)),
                        normalize=True,
                        value_range=self._get_dataset_attr('value_range'),
                    )
                    utils.save_image(
                        grid,
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                    )
                    if self.wandb_cfg is not None:
                        if key in ['sample_cond_shaded', 'image', 'sample_base_color', 'sample_gt_base_color', 'renderer_consistency']:
                            import wandb
                            wandb_log_data[f'val/samples/{key}'] = wandb.Image(grid)
                elif samples[key]['type'] == 'number':
                    min_val = samples[key]['value'].min()
                    max_val = samples[key]['value'].max()
                    images = (samples[key]['value'] - min_val) / (max_val - min_val)
                    grid = utils.make_grid(
                        images,
                        nrow=int(np.sqrt(num_samples)),
                        normalize=False,
                    )
                    save_image_with_notes(
                        grid,
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        notes=f'{key} min: {min_val}, max: {max_val}',
                    )
                    if self.wandb_cfg is not None:
                        import wandb
                        wandb_log_data[f'val/samples/{key}'] = wandb.Image(grid, caption=f'min: {min_val}, max: {max_val}')

            if self.wandb_cfg is not None and wandb_log_data:
                import wandb
                wandb.log(wandb_log_data, step=self.step)

        if self.is_master:
            print(' Done.')

    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        assert self.is_master, 'update_ema() should be called only by the rank 0 process.'
        for i, ema_rate in enumerate(self.ema_rate):
            for master_param, ema_param in zip(self.master_params, self.ema_params[i]):
                ema_param.detach().mul_(ema_rate).add_(master_param, alpha=1.0 - ema_rate)

    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        if self.is_master:
            print('\nPerforming DDP check...')

        if self.is_master:
            print('Checking if parameters are consistent across processes...')
        dist.barrier()
        try:
            for p in self.master_params:
                # split to avoid OOM
                for i in range(0, p.numel(), 10000000):
                    sub_size = min(10000000, p.numel() - i)
                    sub_p = p.detach().view(-1)[i:i+sub_size]
                    # gather from all processes
                    sub_p_gather = [torch.empty_like(sub_p) for _ in range(self.world_size)]
                    dist.all_gather(sub_p_gather, sub_p)
                    # check if equal
                    assert all([torch.equal(sub_p, sub_p_gather[i]) for i in range(self.world_size)]), 'parameters are not consistent across processes'
        except AssertionError as e:
            if self.is_master:
                print(f'\n\033[91mError: {e}\033[0m')
                print('DDP check failed.')
            raise e

        dist.barrier()
        if self.is_master:
            print('Done.')

    @abstractmethod
    def training_losses(**mb_data):
        """
        Compute training losses.
        """
        pass

    def load_data(self):
        """
        Load data.
        """
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        
        # if the data is a dict, we need to split it into multiple dicts with batch_size_per_gpu
        if isinstance(data, dict):
            if self.batch_split == 1:
                data_list = [data]
            else:
                batch_size = list(data.values())[0].shape[0]
                data_list = [
                    {k: v[i * batch_size // self.batch_split:(i + 1) * batch_size // self.batch_split] for k, v in data.items()}
                    for i in range(self.batch_split)
                ]
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError('Data must be a dict or a list of dicts.')
        
        return data_list

    def run_step(self, data_list):
        """
        Run a training step.
        """
        step_log = {'loss': {}, 'status': {}}
        amp_context = partial(torch.autocast, device_type='cuda', dtype=self.mix_precision_dtype) if self.mix_precision_mode == 'amp' else nullcontext
        elastic_controller_context = self.elastic_controller.record if self.elastic_controller_config is not None else nullcontext

        # Train
        losses = []
        statuses = []
        elastic_controller_logs = []
        zero_grad(self.model_params)
        for i, mb_data in enumerate(data_list):
            with torch.no_grad(), amp_context():
                prep_result = self.prep_for_training_iter(**mb_data)
                ext_cond, t, noise, concat_cond = prep_result[:4]
                mb_data['ext_cond'] = ext_cond
                mb_data['t'] = t
                mb_data['noise'] = noise
                mb_data['concat_cond'] = concat_cond
                if len(prep_result) > 4:
                    mb_data.update(prep_result[4])
            
            ## sync at the end of each batch split
            sync_contexts = [self.training_models[name].no_sync for name in self.training_models] if i != len(data_list) - 1 and self.world_size > 1 else [nullcontext]
            with nested_contexts(*sync_contexts), elastic_controller_context():
                with amp_context():
                    loss, status = self.training_losses(**mb_data)
                    l = loss['loss'] / len(data_list)
                ## backward
                if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
                    self.scaler.scale(l).backward()
                elif self.mix_precision_mode == 'inflat_all' and self.mix_precision_dtype == torch.float16:
                    scaled_l = l * (2 ** self.log_scale)
                    scaled_l.backward()
                else:
                    l.backward()
            ## log
            losses.append(dict_foreach(loss, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            statuses.append(dict_foreach(status, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            if self.elastic_controller_config is not None:
                elastic_controller_logs.append(self.elastic_controller.log())
        ## gradient clip
        if self.grad_clip is not None:
            if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
                self.scaler.unscale_(self.optimizer)
            elif self.mix_precision_mode == 'inflat_all':
                model_grads_to_master_grads(self.model_params, self.master_params)
                if self.mix_precision_dtype == torch.float16:
                    self.master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))
            if isinstance(self.grad_clip, float):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params, self.grad_clip)
            else:
                grad_norm = self.grad_clip(self.master_params)
            if torch.isfinite(grad_norm):
                statuses[-1]['grad_norm'] = grad_norm.item()
        ## step
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.mix_precision_mode == 'inflat_all':
            if self.mix_precision_dtype == torch.float16:
                prev_scale = 2 ** self.log_scale
                if not any(not p.grad.isfinite().all() for p in self.model_params):
                    if self.grad_clip is None:
                        model_grads_to_master_grads(self.model_params, self.master_params)
                        self.master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))
                    self.optimizer.step()
                    master_params_to_model_params(self.model_params, self.master_params)
                    self.log_scale += self.fp16_scale_growth
                else:
                    self.log_scale -= 1
            else:
                prev_scale = 1.0
                if self.grad_clip is None:
                    model_grads_to_master_grads(self.model_params, self.master_params)
                if not any(p.grad is not None and not p.grad.isfinite().all() for p in self.master_params):
                    self.optimizer.step()
                    master_params_to_model_params(self.model_params, self.master_params)
                else:
                    print('\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m')
        else:
            prev_scale = 1.0
            if not any(p.grad is not None and not p.grad.isfinite().all() for p in self.model_params):
                self.optimizer.step()
            else:
                print('\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m')
        ## adjust learning rate
        if self.lr_scheduler_config is not None:
            statuses[-1]['lr'] = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()

        # Logs
        step_log['loss'] = dict_reduce(losses, lambda x: np.mean(x))
        step_log['status'] = dict_reduce(statuses, lambda x: np.mean(x), special_func={'min': lambda x: np.min(x), 'max': lambda x: np.max(x)})
        if self.elastic_controller_config is not None:
            step_log['elastic'] = dict_reduce(elastic_controller_logs, lambda x: np.mean(x))
        if self.grad_clip is not None:
            step_log['grad_clip'] = self.grad_clip if isinstance(self.grad_clip, float) else self.grad_clip.log()
            
        # Check grad and norm of each param
        if self.log_param_stats:
            param_norms = {}
            param_grads = {}
            for model_name, model in self.models.items():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param_norms[f'{model_name}.{name}'] = param.norm().item()
                        if param.grad is not None and torch.isfinite(param.grad).all():
                            param_grads[f'{model_name}.{name}'] = param.grad.norm().item() / prev_scale
            step_log['param_norms'] = param_norms
            step_log['param_grads'] = param_grads

        # Update exponential moving average
        if self.is_master:
            self.update_ema()

            # DEBUG: Check if input layer new weights are training
            # if self.mode == 'with_grad' and self.step % 10 == 0:
            #     if 'denoiser' in self.models:
            #         model = self.models['denoiser']
            #         if hasattr(model, 'module'): model = model.module
            #         if hasattr(model, 'input_layer'):
            #             layer = model.input_layer
            #             if hasattr(layer, 'base_layer'): layer = layer.base_layer
            #             if hasattr(layer, 'weight'):
            #                 w = layer.weight
            #                 # Assuming input is [x, concat_cond, grad] -> 3 parts. New weights are the last part.
            #                 latent_dim = w.shape[1] // 3
            #                 new_weights = w[:, -latent_dim:]
            #                 print(f"\n[DEBUG Step {self.step}] Input layer new weights (grad input) stats:")
            #                 print(f"  Weight Mean: {new_weights.mean().item():.6e}, Std: {new_weights.std().item():.6e}, MaxAbs: {new_weights.abs().max().item():.6e}")
            #                 if w.grad is not None:
            #                     new_grads = w.grad[:, -latent_dim:]
            #                     print(f"  Grad Mean:   {new_grads.mean().item():.6e}, Std: {new_grads.std().item():.6e}, MaxAbs: {new_grads.abs().max().item():.6e}")
            #                 else:
            #                     print("  Grad: None")

        return step_log

    def save_logs(self):
        log_str = '\n'.join([
            f'{step}: {json.dumps(dict_foreach(log, lambda x: float(x)))}' for step, log in self.log
        ])
        with open(os.path.join(self.output_dir, 'log.txt'), 'a') as log_file:
            log_file.write(log_str + '\n')

        # show with mlflow
        if self.wandb_cfg is not None:
            pass
        else:
            log_show = [l for _, l in self.log if not dict_any(l, lambda x: np.isnan(x))]
            log_show = dict_reduce(log_show, lambda x: np.mean(x))
            log_show = dict_flatten(log_show, sep='/')
            for key, value in log_show.items():
                self.writer.add_scalar(key, value, self.step)
        self.log = []
        
    def check_abort(self):
        """
        Check if training should be aborted due to certain conditions.
        """
        # 1. If log_scale in inflat_all mode is less than 0
        if self.mix_precision_dtype == torch.float16 and \
           self.mix_precision_mode == 'inflat_all' and \
           self.log_scale < 0:
            if self.is_master:
                print ('\n\n\033[91m')
                print (f'ABORT: log_scale in inflat_all mode is less than 0 at step {self.step}.')
                print ('This indicates that the model is diverging. You should look into the model and the data.')
                print ('\033[0m')
                self.save(non_blocking=False)
                self.save_logs()
            if self.world_size > 1:
                dist.barrier()
            raise ValueError('ABORT: log_scale in inflat_all mode is less than 0.')

    def run(self):
        """
        Run training.
        """
        if self.is_master:
            print('\nStarting training...')
            self.snapshot_dataset(num_samples=2, batch_size=self.snapshot_batch_size)
        if self.step == 0:
            self.snapshot(suffix='init', num_samples=10, batch_size=self.snapshot_batch_size)
        else: # resume
            self.snapshot(suffix=f'resume_step{self.step:07d}', num_samples=10, batch_size=self.snapshot_batch_size)

        time_last_print = 0.0
        time_elapsed = 0.0
        while self.step < self.max_steps:
            time_start = time.time()

            data_list = self.load_data()
            step_log = self.run_step(data_list)

            time_end = time.time()
            time_elapsed += time_end - time_start

            self.step += 1

            # Print progress
            if self.is_master and self.step % self.i_print == 0:
                speed = self.i_print / (time_elapsed - time_last_print) * 3600
                columns = [
                    f'Step: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)',
                    f'Elapsed: {time_elapsed / 3600:.2f} h',
                    f'Speed: {speed:.2f} steps/h',
                    f'ETA: {(self.max_steps - self.step) / speed:.2f} h',
                ]
                print(' | '.join([c.ljust(25) for c in columns]), flush=True)
                time_last_print = time_elapsed

            # Check ddp
            if self.parallel_mode == 'ddp' and self.world_size > 1 and self.i_ddpcheck is not None and self.step % self.i_ddpcheck == 0:
                self.check_ddp()

            # Sample images
            if self.step % self.i_sample == 0:
                if self.max_samples is not None:
                    num_samples = min(self.max_samples, 10)
                else:
                    num_samples = 10
                self.snapshot(num_samples=num_samples, batch_size=self.snapshot_batch_size)

            if self.is_master:
                self.log.append((self.step, {}))

                # Log time
                self.log[-1][1]['time'] = {
                    'step': time_end - time_start,
                    'elapsed': time_elapsed,
                }

                # Log losses
                if step_log is not None:
                    self.log[-1][1].update(step_log)

                # Log scale
                if self.mix_precision_dtype == torch.float16:
                    if self.mix_precision_mode == 'amp':
                        self.log[-1][1]['scale'] = self.scaler.get_scale()
                    elif self.mix_precision_mode == 'inflat_all':
                        self.log[-1][1]['log_scale'] = self.log_scale

                if self.wandb_cfg is not None:
                    import wandb
                    current_log = self.log[-1][1]
                    if not dict_any(current_log, lambda x: np.isnan(x)):
                        wandb.log(dict_flatten(current_log, sep='/'), step=self.step)

                # Save log
                if self.step % self.i_log == 0:
                    self.save_logs()

                # Save checkpoint
                if self.step % self.i_save == 0:
                    self.save()
                    
            # Check abort
            self.check_abort()

        if self.max_samples is not None:
            num_final_samples = min(self.max_samples, 10)
        else:
            num_final_samples = 10
        self.snapshot(suffix='final', num_samples=num_final_samples, batch_size=self.snapshot_batch_size)
        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            if self.wandb_cfg is not None:
                import wandb
                wandb.finish()
            else:
                self.writer.close()
            print('Training finished.')
            
    def profile(self, wait=2, warmup=3, active=5):
        """
        Profile the training loop.
        """
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, 'profile')),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                self.run_step()
                prof.step()
