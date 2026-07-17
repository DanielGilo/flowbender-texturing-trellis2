import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import json
from typing import *
import numpy as np
import torch
from .. import models
from .components import StandardDatasetBase, ImageConditionedMixin
from ..modules.sparse import SparseTensor, sparse_cat
from ..representations import MeshWithVoxel
import cv2
from ..renderers import PbrMeshRenderer, EnvMap
from ..utils.data_utils import load_balanced_group_indices
from ..utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics


class SLatPbrVisMixin:
    def __init__(
        self,
        *args,
        pretrained_pbr_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16',
        pbr_slat_dec_path: Optional[str] = None,
        pbr_slat_dec_ckpt: Optional[str] = None,
        pretrained_shape_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        shape_slat_dec_path: Optional[str] = None,
        shape_slat_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pbr_slat_dec = None
        self.pretrained_pbr_slat_dec = pretrained_pbr_slat_dec
        self.pbr_slat_dec_path = pbr_slat_dec_path
        self.pbr_slat_dec_ckpt = pbr_slat_dec_ckpt
        self.shape_slat_dec = None
        self.pretrained_shape_slat_dec = pretrained_shape_slat_dec
        self.shape_slat_dec_path = shape_slat_dec_path
        self.shape_slat_dec_ckpt = shape_slat_dec_ckpt
        
    def _loading_slat_dec(self):
        if self.pbr_slat_dec is not None and self.shape_slat_dec is not None:
            return
        if self.pbr_slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.pbr_slat_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.pbr_slat_dec_path, 'ckpts', f'decoder_{self.pbr_slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_pbr_slat_dec)
        self.pbr_slat_dec = decoder.cuda().eval()
        self.pbr_slat_dec.requires_grad_(False)

        if self.shape_slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.shape_slat_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.shape_slat_dec_path, 'ckpts', f'decoder_{self.shape_slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_shape_slat_dec)
        decoder.set_resolution(self.resolution)
        self.shape_slat_dec = decoder.cuda().eval()
        self.shape_slat_dec.requires_grad_(False)

    def _delete_slat_dec(self):
        del self.pbr_slat_dec
        self.pbr_slat_dec = None
        del self.shape_slat_dec
        self.shape_slat_dec = None
        
    def decode_latent(self, z, shape_z, batch_size=4):
        self._loading_slat_dec()
        reps = []
        if self.shape_slat_normalization is not None:
            shape_z = shape_z * self.shape_slat_std.to(z.device) + self.shape_slat_mean.to(z.device)
        if self.pbr_slat_normalization is not None:
            z = z * self.pbr_slat_std.to(z.device) + self.pbr_slat_mean.to(z.device)
            
        # Ensure inputs match decoder dtype
        dtype = next(self.shape_slat_dec.parameters()).dtype
        shape_z = shape_z.type(dtype)
        
        dtype = next(self.pbr_slat_dec.parameters()).dtype
        z = z.type(dtype)

        for i in range(0, z.shape[0], batch_size):
            mesh, subs = self.shape_slat_dec(shape_z[i:i+batch_size], return_subs=True)
            vox = self.pbr_slat_dec(z[i:i+batch_size], guide_subs=subs) * 0.5 + 0.5
            reps.extend([
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / self.resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout = self.layout,
                )
                for m, v in zip(mesh, vox)
            ])
        # Do NOT delete the decoders! We need them permanently cached
        # in VRAM for the continuous rendering loss loop.
        return reps
    
    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        shape_z = sample['concat_cond'].cuda()
        z = sample['x_0'].cuda()
        reps = self.decode_latent(z, shape_z, batch_size=1)  # small batch to prevent OOM
        
        # build camera
        yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
        yaw_offset = -16 / 180 * np.pi
        yaw = [y + yaw_offset for y in yaw]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)
        
        # render
        renderer = PbrMeshRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 1
        renderer.rendering_options.far = 100
        renderer.rendering_options.ssaa = 2
        renderer.rendering_options.peel_layers = 8
        
        # Use batch envmaps if available, otherwise fallback to static forest
        use_dynamic_envmap = 'cond_envmap' in sample
        if not use_dynamic_envmap and not hasattr(self, '_static_envmap'):
            self._static_envmap = EnvMap(torch.tensor(
                cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                dtype=torch.float32, device='cuda'
            ))
        
        images = {}
        for i, representation in enumerate(reps):
            image = {}
            tile = [2, 2]
            
            current_envmap = EnvMap(sample['cond_envmap'][i].cuda()) if use_dynamic_envmap else self._static_envmap

            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, envmap=current_envmap)
                for k, v in res.items():
                    if k not in images:
                        images[k] = []
                    if k not in image:
                        image[k] = torch.zeros(3, 1024, 1024).cuda()  
                    image[k][:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = v
            for k in images.keys():
                images[k].append(image[k])
        for k in images.keys():
            images[k] = torch.stack(images[k], dim=0)
        return images
    
    @torch.no_grad()
    def visualize_cond_sample(self, sample: dict):
        if 'cond_extrinsics' not in sample or 'cond_intrinsics' not in sample:
            return {}
        
        shape_z = sample['concat_cond'].cuda()
        z = sample['x_0'].cuda()
        reps = self.decode_latent(z, shape_z, batch_size=1)
        
        cond_extrinsics = sample['cond_extrinsics'].cuda()
        cond_intrinsics = sample['cond_intrinsics'].cuda()     
        
        # render
        renderer = PbrMeshRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 1 #0.01
        renderer.rendering_options.far = 100
        renderer.rendering_options.ssaa = 2
        renderer.rendering_options.peel_layers = 8

        # Check if conditional envmaps are provided in the batch
        use_dynamic_envmap = 'cond_envmap' in sample
        if not use_dynamic_envmap:
            # Fallback to the static forest envmap if not provided
            # Cache it to avoid reloading
            if not hasattr(self, '_static_envmap'):
                self._static_envmap = EnvMap(torch.tensor(
                    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                    dtype=torch.float32, device='cuda'
                ))
        
        images = {}
        for i, representation in enumerate(reps):
            # Select the correct envmap for this item in the batch
            if use_dynamic_envmap:
                # Create an EnvMap on the fly from the batched tensor
                current_envmap = EnvMap(sample['cond_envmap'][i].cuda())
            else:
                # Use the cached static envmap
                current_envmap = self._static_envmap
            res = renderer.render(representation, cond_extrinsics[i], cond_intrinsics[i], envmap=current_envmap)
            for k, v in res.items():
                if k != 'shaded':
                    continue
                if k not in images:
                    images[k] = []
                v = v * (res['alpha'].clamp(0, 1))
                images[k].append(v)
        for k in images.keys():
            images[k] = torch.stack(images[k], dim=0)
        return images
    
    @torch.no_grad()
    def check_renderer_consistency(self, sample: dict):
        """
        Debug function to compare GT image with Renderer output of GT latent.
        Returns a tensor of images: [GT_Image | Rendered_GT_Latent]
        """
        if 'cond_extrinsics' not in sample or 'cond_intrinsics' not in sample or 'cond' not in sample:
            return None
        
        shape_z = sample['concat_cond'].cuda()
        z = sample['x_0'].cuda() # GT PBR Latent
        
        with torch.autocast(device_type='cuda', enabled=False):
            # 1. Decode GT Latents
            reps = self.decode_latent(z.float(), shape_z.float(), batch_size=1)
            
            cond_extrinsics = sample['cond_extrinsics'].cuda().float()
            cond_intrinsics = sample['cond_intrinsics'].cuda().float()     
            
            # 2. Setup Renderer
            renderer = PbrMeshRenderer()
            renderer.rendering_options.resolution = 512
            renderer.rendering_options.near = 1 # 0.01
            renderer.rendering_options.far = 100
            renderer.rendering_options.ssaa = 2
            renderer.rendering_options.peel_layers = 8

            use_dynamic_envmap = 'cond_envmap' in sample
            if not use_dynamic_envmap and not hasattr(self, '_static_envmap'):
                self._static_envmap = EnvMap(torch.tensor(
                    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                    dtype=torch.float32, device='cuda'
                ))
            
            images = []
            for i, representation in enumerate(reps):
                # 3. Match Environment
                current_envmap = EnvMap(sample['cond_envmap'][i].cuda().float()) if use_dynamic_envmap else self._static_envmap
                
                # 4. Render from Condition View
                res = renderer.render(representation, cond_extrinsics[i], cond_intrinsics[i], envmap=current_envmap)
                
                # 5. Prepare GT Image
                gt_img = sample['cond'][i].cuda().float() # [3, H, W]
                
                
                render_img = res['shaded'].clamp(0, 1)
                render_alpha = res['alpha'].clamp(0, 1)
                if render_alpha.dim() == 2:
                    render_alpha = render_alpha.unsqueeze(0)
                
                render_img = render_img * render_alpha
                # 6. Resize Render if necessary (to match GT)
                if render_img.shape[-1] != gt_img.shape[-1]:
                    print("WARNING: resizing rendered image to match gt!")
                    render_img = F.interpolate(render_img.unsqueeze(0), size=gt_img.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

                # 7. Concatenate Side-by-Side [GT | Render]
                images.append(torch.cat([gt_img, render_img], dim=2))
                
            return torch.stack(images, dim=0)
    
    def __getstate__(self):
        # This tells Python how to pickle the dataset for multiprocessing workers.
        # We strip out the un-picklable PyTorch models. The workers will just 
        # lazy-load them again via _loading_slat_dec() if they ever need them.
        state = self.__dict__.copy()
        state['pbr_slat_dec'] = None
        state['shape_slat_dec'] = None
        
        if '_static_envmap' in state:
            state['_static_envmap'] = None
        return state
    
    
class SLatPbr(SLatPbrVisMixin, StandardDatasetBase):
    """
    structured latent for sparse voxel pbr dataset
    
    Args:
        roots (str): path to the dataset
        latent_key (str): key of the latent to be used
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        resolution (int): resolution of decoded sparse voxel
        attrs (list): attributes to be decoded
        pretained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        resolution: int,
        min_aesthetic_score: float = 5.0,
        max_tokens: int = 32768,
        full_pbr: bool = False,
        pbr_slat_normalization: Optional[dict] = None,
        shape_slat_normalization: Optional[dict] = None,
        attrs: list[str] = ['base_color', 'metallic', 'roughness', 'emissive', 'alpha'],
        pretrained_pbr_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16',
        pbr_slat_dec_path: Optional[str] = None,
        pbr_slat_dec_ckpt: Optional[str] = None,
        pretrained_shape_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        shape_slat_dec_path: Optional[str] = None,
        shape_slat_dec_ckpt: Optional[str] = None,
        **kwargs
    ):  
        self.resolution = resolution
        self.pbr_slat_normalization = pbr_slat_normalization
        self.shape_slat_normalization = shape_slat_normalization
        self.min_aesthetic_score = min_aesthetic_score
        self.max_tokens = max_tokens
        self.full_pbr = full_pbr
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_pbr_slat_dec=pretrained_pbr_slat_dec,
            pbr_slat_dec_path=pbr_slat_dec_path,
            pbr_slat_dec_ckpt=pbr_slat_dec_ckpt,
            pretrained_shape_slat_dec=pretrained_shape_slat_dec,
            shape_slat_dec_path=shape_slat_dec_path,
            shape_slat_dec_ckpt=shape_slat_dec_ckpt,
            **kwargs
        )
        
        self.loads = [self.metadata.loc[sha256, 'pbr_latent_tokens'] for _, sha256 in self.instances]
        
        if self.pbr_slat_normalization is not None:
            self.pbr_slat_mean = torch.tensor(self.pbr_slat_normalization['mean']).reshape(1, -1)
            self.pbr_slat_std = torch.tensor(self.pbr_slat_normalization['std']).reshape(1, -1)
        
        if self.shape_slat_normalization is not None:
            self.shape_slat_mean = torch.tensor(self.shape_slat_normalization['mean']).reshape(1, -1)
            self.shape_slat_std = torch.tensor(self.shape_slat_normalization['std']).reshape(1, -1)
        
        self.attrs = attrs
        self.channels = {
            'base_color': 3,
            'metallic': 1,
            'roughness': 1,
            'emissive': 3,
            'alpha': 1,
        }
        self.layout = {}
        start = 0
        for attr in attrs:
            self.layout[attr] = slice(start, start + self.channels[attr])
            start += self.channels[attr]
            
    def filter_metadata(self, metadata):
        stats = {}
        metadata = metadata[metadata['pbr_latent_encoded'] == True]
        stats['With PBR latent'] = len(metadata)
        metadata = metadata[metadata['shape_latent_encoded'] == True]
        stats['With shape latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['pbr_latent_tokens'] <= self.max_tokens]
        stats[f'Num tokens <= {self.max_tokens}'] = len(metadata)
        if self.full_pbr:
            metadata = metadata[metadata['num_basecolor_tex'] > 0]
            metadata = metadata[metadata['num_metallic_tex'] > 0]
            metadata = metadata[metadata['num_roughness_tex'] > 0]
            stats['Full PBR'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        # PBR latent
        data = np.load(os.path.join(root['pbr_latent'], f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        coords = torch.cat([torch.zeros_like(coords)[:, :1], coords], dim=1)
        feats = torch.tensor(data['feats']).float()
        if self.pbr_slat_normalization is not None:
            feats = (feats - self.pbr_slat_mean) / self.pbr_slat_std
        pbr_z = SparseTensor(feats, coords)
        
        # Shape latent
        data = np.load(os.path.join(root['shape_latent'], f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        coords = torch.cat([torch.zeros_like(coords)[:, :1], coords], dim=1)
        feats = torch.tensor(data['feats']).float()
        if self.shape_slat_normalization is not None:
            feats = (feats - self.shape_slat_mean) / self.shape_slat_std
        shape_z = SparseTensor(feats, coords)
        
        assert torch.equal(shape_z.coords, pbr_z.coords), \
            f"Shape latent and PBR latent have different coordinates: {shape_z.coords.shape} vs {pbr_z.coords.shape}"
            
        return {
            'x_0': pbr_z,
            'concat_cond': shape_z,
        }
        
    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['x_0'].feats.shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}

            keys = [k for k in sub_batch[0].keys()]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], SparseTensor):
                    pack[k] = sparse_cat([b[k] for b in sub_batch], dim=0)
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
            
            packs.append(pack)
        
        if split_size is None:
            return packs[0]
        return packs


class ImageConditionedSLatPbr(ImageConditionedMixin, SLatPbr):
    """
    Image conditioned structured latent dataset
    """
    pass
