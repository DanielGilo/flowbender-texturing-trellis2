import os
import sys

# Enable EXR loading for OpenCV
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import json
import argparse
import pandas as pd
import numpy as np
import cv2
import torch
from tqdm import tqdm
from easydict import EasyDict as edict

# Import TRELLIS modules
from trellis2 import models
from trellis2.modules.sparse import SparseTensor
from trellis2.representations.mesh import MeshWithVoxel
from trellis2.renderers import PbrMeshRenderer, EnvMap
import utils3d

from utils import sphere_hammersley_sequence

def load_latent(path: str) -> SparseTensor:
    """Loads a sparse latent from an npz file."""
    data = np.load(path)
    coords = torch.tensor(data['coords']).int()
    coords = torch.cat([torch.zeros_like(coords)[:, :1], coords], dim=1)
    feats = torch.tensor(data['feats']).float()
    return SparseTensor(feats, coords).cuda()

def main(opt):
    os.makedirs(os.path.join(opt.render_cond_root, 'renders_cond'), exist_ok=True)
    os.makedirs(os.path.join(opt.render_cond_root, 'renders_cond', 'new_records'), exist_ok=True)
    
    print('Loading metadata...', flush=True)
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256').reset_index()
    
    if opt.instances is not None:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    pbr_latent_dir = os.path.join(opt.root, 'pbr_latents', 'tex_enc_next_dc_f16c32_fp16_512')
    shape_latent_dir = os.path.join(opt.root, 'shape_latents', 'shape_enc_next_dc_f16c32_fp16_512')
    
    records = []
    missing_instances = []
    for sha256 in metadata['sha256'].values:
        if not (os.path.exists(os.path.join(pbr_latent_dir, f'{sha256}.npz')) and 
                os.path.exists(os.path.join(shape_latent_dir, f'{sha256}.npz'))):
            continue
        if not os.path.exists(os.path.join(opt.render_cond_root, 'renders_cond', sha256, 'transforms.json')):
            missing_instances.append(sha256)
        else:
            records.append({'sha256': sha256, 'cond_rendered': True})
    
    if not missing_instances:
        if records:
            print("All existing latents have already been rendered. Writing records and exiting.")
            pd.DataFrame.from_records(records).to_csv(os.path.join(opt.render_cond_root, 'renders_cond', 'new_records', f'part_{opt.rank}.csv'), index=False)
        else:
            print("No latents available to render. Exiting.")
        return
        
    print(f'Found {len(missing_instances)} objects ready for rendering...')

    print("Loading TRELLIS VAE Decoders...", flush=True)
    shape_dec = models.from_pretrained('JeffreyXiang/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16').cuda().eval()
    shape_dec.set_resolution(512)
    shape_dec.requires_grad_(False)
    
    pbr_dec = models.from_pretrained('JeffreyXiang/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16').cuda().eval()
    pbr_dec.requires_grad_(False)
    
    # STRICT REPLICATION: Exact layout generation logic from SLatPbr.__init__
    layout_attrs = ['base_color', 'metallic', 'roughness', 'alpha']
    channels_dict = {'base_color': 3, 'metallic': 1, 'roughness': 1, 'emissive': 3, 'alpha': 1}
    layout = {}
    start_idx = 0
    for attr in layout_attrs:
        layout[attr] = slice(start_idx, start_idx + channels_dict[attr])
        start_idx += channels_dict[attr]

    print("Initializing PBR Renderer...", flush=True)
    renderer = PbrMeshRenderer()
    renderer.rendering_options.resolution = opt.cond_resolution
    renderer.rendering_options.near = 1.0
    renderer.rendering_options.far = 100.0
    renderer.rendering_options.ssaa = 2
    renderer.rendering_options.peel_layers = 8

    hdri_path = os.path.abspath('assets/hdri/forest.exr')
    static_envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread(hdri_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device='cuda'
    ))

    for sha256 in tqdm(missing_instances, desc="Rendering PBR Conditions"):
        out_folder = os.path.join(opt.render_cond_root, 'renders_cond', sha256)
        os.makedirs(out_folder, exist_ok=True)
        
        with torch.autocast(device_type='cuda', enabled=False):
            shape_z = load_latent(os.path.join(shape_latent_dir, f'{sha256}.npz'))
            pbr_z = load_latent(os.path.join(pbr_latent_dir, f'{sha256}.npz'))
            
            # Match decode_latent dtype casting
            dtype_shape = next(shape_dec.parameters()).dtype
            shape_z = shape_z.type(dtype_shape)
            
            dtype_pbr = next(pbr_dec.parameters()).dtype
            pbr_z = pbr_z.type(dtype_pbr)

            with torch.no_grad():
                mesh_list, subs = shape_dec(shape_z, return_subs=True)
                vox_list = pbr_dec(pbr_z, guide_subs=subs) * 0.5 + 0.5
                
                # REPLICATION: Use zip loop to correctly handle SparseTensor batch components
                reps = []
                for m, v in zip(mesh_list, vox_list):
                    reps.append(MeshWithVoxel(
                        m.vertices.float(), m.faces,
                        origin=[-0.5, -0.5, -0.5],
                        voxel_size=1 / 512, 
                        coords=v.coords[:, 1:],
                        attrs=v.feats.float(),
                        voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
                        layout=layout,
                    ))
                representation = reps[0]
            
            # --- STRICT REPLICATION: Use Trellis's native batched camera utility ---
            yaws, pitchs = [], []
            offset = (np.random.rand(), np.random.rand())
            
            for i in range(opt.num_cond_views):
                y, p = sphere_hammersley_sequence(i, opt.num_cond_views, offset)
                yaws.append(y)
                pitchs.append(p)
                
            fov_min, fov_max = 10, 70
            radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
            radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
            ks = np.random.uniform(1 / radius_max**2, 1 / radius_min**2, (opt.num_cond_views,))
            
            radius_list = [1 / np.sqrt(k) for k in ks]
            fovs = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius_list]
            
            # Explicitly compute extrinsics and intrinsics as expected by the PbrMeshRenderer
            exts, ints = [], []
            for y, p, r, f in zip(yaws, pitchs, radius_list, fovs):
                orig = torch.tensor([
                    np.cos(y) * np.cos(p),
                    np.sin(y) * np.cos(p),
                    np.sin(p),
                ]).float().cuda() * r
                ext = utils3d.torch.extrinsics_look_at(
                    orig, 
                    torch.tensor([0, 0, 0]).float().cuda(), 
                    torch.tensor([0, 0, 1]).float().cuda()
                )
                intr = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(f).cuda(), torch.tensor(f).cuda())
                exts.append(ext)
                ints.append(intr)

            to_export = {
                "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                "scale": 1.0, 
                "offset": [0.0, 0.0, 0.0],
                "frames": []
            }

            with torch.no_grad():
                for i, (ext, intr) in enumerate(zip(exts, ints)):
                    # Keep it inside the disabled autocast block, exactly as in check_renderer_consistency
                    res = renderer.render(representation, ext.cuda().float(), intr.cuda().float(), envmap=static_envmap)
                    
                    shaded_img = res['shaded'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    
                    # alpha is [H, W] because the renderer applies .squeeze() which removes the 1-channel dim
                    alpha_tensor = res['alpha'].clamp(0, 1)
                    if alpha_tensor.dim() == 2:
                        alpha_img = alpha_tensor.unsqueeze(-1).cpu().numpy()
                    else:
                        alpha_img = alpha_tensor.permute(1, 2, 0).cpu().numpy()
                        
                    # Un-premultiply the RGB to get straight RGB, matching standard PNG output
                    # This prevents the dataset loader from double-multiplying the alpha
                    straight_shaded_img = np.clip(shaded_img / (alpha_img + 1e-6), 0, 1)
                    
                    # Combine to RGBA to preserve the alpha mask, essential for the conditional diffusion models
                    rgba_img = np.concatenate([straight_shaded_img, alpha_img], axis=-1)
                    rgba_img = (rgba_img * 255).astype(np.uint8)
                    bgra_img = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGRA)
                    
                    file_name = f'{i:03d}.png'
                    cv2.imwrite(os.path.join(out_folder, file_name), bgra_img)
                    
                    # IMPORTANT: The Dataset (ImageConditionedMixin) expects a Blender-style transform matrix.
                    # It reads T, computes E = inv(T), and flips Y and Z: E[1:3, :] *= -1.
                    # To ensure it recovers our exact PyTorch3D `ext` matrix, we must reverse this process before saving.
                    E_to_save = ext.clone()
                    E_to_save[1:3, :] *= -1
                    transform_matrix = torch.linalg.inv(E_to_save).cpu().numpy().tolist()
                    
                    to_export["frames"].append({
                        "file_path": file_name,
                        "envmap_path": "forest.exr",
                        "camera_angle_x": fovs[i],
                        "transform_matrix": transform_matrix
                    })

            with open(os.path.join(out_folder, 'transforms.json'), 'w') as f:
                json.dump(to_export, f, indent=4)
                    
            records.append({'sha256': sha256, 'cond_rendered': True})

    # Save metadata for this split
    pd.DataFrame.from_records(records).to_csv(os.path.join(opt.render_cond_root, 'renders_cond', 'new_records', f'part_{opt.rank}.csv'), index=False)
    print("Done! Metadata saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--render_cond_root', type=str, default=None)
    parser.add_argument('--instances', type=str, default=None)
    parser.add_argument('--num_cond_views', type=int, default=16)
    parser.add_argument('--cond_resolution', type=int, default=1024)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    
    opt = parser.parse_args()
    opt = edict(vars(opt))
    opt.render_cond_root = opt.render_cond_root or opt.root

    main(opt)