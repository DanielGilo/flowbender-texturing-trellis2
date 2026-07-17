import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'  # needed to load .exr HDRI environment maps
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import json
import argparse
from easydict import EasyDict as edict
from tqdm import tqdm
import cv2
from contextlib import nullcontext
from functools import partial
import gc

import numpy as np
import math
import imageio

import lpips

from trellis2 import models, datasets, trainers
from trellis2.utils.data_utils import recursive_to_device
from trellis2.modules import sparse as sp
from trellis2.renderers import PbrMeshRenderer, EnvMap

from eval_common import find_ckpt, set_seed, get_random_camera, compute_ssim, seed_worker


def evaluate(cfg, num_samples: int = 100, seed: int = 42, gt_fid_dir: str = 'eval_gt_fid_views', grad_cfg_strength: float = 1.0, save_videos: bool = False, video_resolution: int = 512, guidance_strength: float = 1.0, start_idx: int = 0, video_indices: set = None, t_thresh: float = 0.0):
    set_seed(seed)

    use_shortcut = t_thresh > 0.0
    if use_shortcut and grad_cfg_strength != 1.0:
        raise ValueError("--t_thresh < 1.0 is not compatible with --grad_cfg_strength != 1.0: "
                          "shortcut steps have no cached unguided velocity to blend with.")

    # 1. Setup LPIPS
    print("Loading LPIPS VGG model...")
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    # 1.5 Setup CLIP
    print("Loading CLIP model...")
    from transformers import CLIPVisionModelWithProjection
    import torchvision.transforms as T
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").cuda().to(torch.float32)
    clip_model.eval()
    clip_model.requires_grad_(False)
    clip_normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    # 2. Setup Dataset (Override split to test)
    print(f"Loading Dataset {cfg.dataset.name} in 'test' split...")
    cfg.dataset.args.split = 'test'
    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)

    # 3. Build Models
    print("Building models...")
    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda()
        for name, model in cfg.models.items()
    }

    # Disable wandb specifically for evaluation
    cfg.trainer.args['wandb_cfg'] = None

    # 4. Build Trainer (To reuse LoRA injection, ckpt loading, input layer resizing)
    print("Initializing trainer and loading weights...")
    trainer = getattr(trainers, cfg.trainer.name)(
        model_dict,
        dataset,
        **cfg.trainer.args,
        output_dir=cfg.output_dir,
        load_dir=cfg.load_dir,
        step=cfg.load_ckpt
    )

    # Ensure models are in evaluation mode
    trainer.models['denoiser'].eval()

    if use_shortcut and trainer.mode not in ('with_grad', 'with_prox_grad'):
        raise ValueError(f"--t_thresh < 1.0 requires mode in ('with_grad', 'with_prox_grad'), got '{trainer.mode}'")

    g = torch.Generator()
    g.manual_seed(seed)

    if start_idx > 0:
        dataset = torch.utils.data.Subset(dataset, range(start_idx, len(dataset)))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,  # Set to 2 to drastically reduce peak VRAM on complex meshes
        shuffle=False,
        num_workers=4,
        collate_fn=trainer._get_dataset_attr('collate_fn'),
        worker_init_fn=seed_worker,
        generator=g,
    )

    sampler = trainer.get_sampler()

    # 5. Setup Renderer
    renderer = PbrMeshRenderer()
    renderer.rendering_options.resolution = 512
    renderer.rendering_options.near = 1
    renderer.rendering_options.far = 100
    renderer.rendering_options.ssaa = 2
    renderer.rendering_options.peel_layers = 8

    static_envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device='cuda'
    ))

    psnr_list = []
    masked_psnr_list = []
    ssim_list = []
    lpips_list = []
    clip_list = []
    mv_psnr_list = []
    mv_masked_psnr_list = []
    mv_ssim_list = []
    mv_lpips_list = []
    mv_clip_list = []
    count = start_idx      # global sample index — used for naming and dataset access
    processed = 0          # samples processed this run — used for stopping
    results_meta = {"samples": [], "summary": {}}

    num_sampling_steps = cfg.num_steps
    num_fid_views = 50

    decode_latent = trainer._get_dataset_attr('decode_latent')

    gcfg_suffix = f"_gradcfg_{grad_cfg_strength}" if grad_cfg_strength != 1.0 else ""
    tthresh_suffix = f"_tthresh_{t_thresh}" if use_shortcut else ""
    eval_dir = os.path.join(cfg.output_dir, f"eval_results_step_{cfg.load_ckpt}_{num_sampling_steps}_steps_guidance_{guidance_strength}{gcfg_suffix}{tthresh_suffix}")
    os.makedirs(eval_dir, exist_ok=True)

    if save_videos:
        videos_dir = os.path.join(eval_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

    pred_views_dir = os.path.join(eval_dir, "pred_fid_views")
    os.makedirs(pred_views_dir, exist_ok=True)
    os.makedirs(gt_fid_dir, exist_ok=True)

    if trainer.mix_precision_mode == 'amp':
        amp_context = partial(torch.autocast, device_type='cuda', dtype=trainer.mix_precision_dtype)
    else:
        amp_context = nullcontext

    mode_info = f"'{trainer.mode}' mode"

    print(f"Starting evaluation in {mode_info}...")
    print(f"Saving results to {eval_dir}...")
    with torch.no_grad(), tqdm(total=num_samples, desc="Evaluating test set") as pbar:
        for batch_data in dataloader:
            if processed >= num_samples:
                break

            batch_size = len(batch_data['cond'])
            if processed + batch_size > num_samples:
                batch_size = num_samples - processed
                batch_data = {k: v[:batch_size] for k, v in batch_data.items()}

            batch_data = recursive_to_device(batch_data, 'cuda')

            # Isolate variables
            noise = batch_data['x_0'].replace(torch.randn_like(batch_data['x_0'].feats))
            shape_z = batch_data['concat_cond']
            cond_imgs = batch_data['cond'] # GT Image

            args = trainer.get_inference_cond(**batch_data)

            # 6. Mode-specific sampling configuration (unused when use_shortcut: sample_with_prior_step_shortcut below handles it internally)
            cond_update_fn = None
            if use_shortcut:
                pass
            elif trainer.mode == 'with_grad':
                use_grad_cfg = grad_cfg_strength != 1.0
                def update_fn(x_t, t, encoded_cond, current_neg_cond):
                    t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.float32)
                    loss_f = trainer.get_render_loss_fn()
                    result = trainer.get_loss_grad_wrt_x_t(x_t, t_tensor, encoded_cond=encoded_cond, loss_f=loss_f, return_pred_v=use_grad_cfg, **batch_data)
                    if use_grad_cfg:
                        grad, pred_v_unguided = result
                    else:
                        grad, pred_v_unguided = result, None
                    concat_cond = sp.sparse_cat([batch_data['concat_cond'], grad], dim=1)
                    return encoded_cond, current_neg_cond, concat_cond, pred_v_unguided
                cond_update_fn = update_fn

            elif trainer.mode == 'with_prox_grad':
                use_grad_cfg = grad_cfg_strength != 1.0
                def update_fn(x_t, t, encoded_cond, current_neg_cond):
                    t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.float32)
                    loss_f = trainer.get_render_loss_fn()
                    result = trainer.get_loss_grad_wrt_pred_x_0(x_t, t_tensor, encoded_cond=encoded_cond, loss_f=loss_f, return_pred_v=use_grad_cfg, **batch_data)
                    if use_grad_cfg:
                        grad, pred_v_unguided = result
                    else:
                        grad, pred_v_unguided = result, None
                    concat_cond = sp.sparse_cat([batch_data['concat_cond'], grad], dim=1)
                    return encoded_cond, current_neg_cond, concat_cond, pred_v_unguided
                cond_update_fn = update_fn

            # Generate Texture Latent
            with amp_context():
                if use_shortcut:
                    pred_z = trainer.sample_with_prior_step_shortcut(
                        noise=noise,
                        cond=args['cond'],
                        neg_cond=args['neg_cond'],
                        batch_data=batch_data,
                        steps=num_sampling_steps,
                        t_thresh=t_thresh,
                        guidance_strength=guidance_strength,
                    )
                else:
                    res = sampler.sample(
                        trainer.models['denoiser'],
                        noise=noise,
                        **args,
                        cond_update_fn=cond_update_fn,
                        steps=num_sampling_steps,
                        guidance_strength=guidance_strength,
                        grad_cfg_strength=grad_cfg_strength,
                        verbose=False,
                    )
                    pred_z = res.samples

            # Decode to Mesh with Voxels (Using GT Shape Latent)
            reps = decode_latent(pred_z.float(), shape_z.float(), batch_size=1)

            # Decode GT mesh when needed (FID views not yet cached)
            gt_needs_render = any(
                not os.path.exists(os.path.join(gt_fid_dir, f"sample_{count + i:04d}_v{v:02d}.png"))
                for i in range(len(reps))
                for v in range(num_fid_views)
            )
            if gt_needs_render:
                reps_gt = decode_latent(batch_data['x_0'].float(), shape_z.float(), batch_size=1)

            cond_extrinsics = batch_data['cond_extrinsics'].float()
            cond_intrinsics = batch_data['cond_intrinsics'].float()
            use_dynamic_envmap = 'cond_envmap' in batch_data

            # 7. Render and compare
            for i, rep in enumerate(reps):
                current_envmap = EnvMap(batch_data['cond_envmap'][i].float()) if use_dynamic_envmap else static_envmap

                try:
                    sample_name = f"sample_{count + i:04d}"

                    render_res = renderer.render(rep, cond_extrinsics[i], cond_intrinsics[i], envmap=current_envmap)
                    render_img = render_res['shaded'].clamp(0, 1)
                    render_alpha = render_res['alpha'].clamp(0, 1)
                    if render_alpha.dim() == 2:
                        render_alpha = render_alpha.unsqueeze(0)

                    render_img = render_img * render_alpha
                    gt_img = cond_imgs[i].float()

                    if render_img.shape[-1] != gt_img.shape[-1]:
                        render_img = F.interpolate(render_img.unsqueeze(0), size=gt_img.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

                    # PSNR
                    mse = torch.mean((gt_img - render_img) ** 2)
                    psnr = -10 * torch.log10(mse + 1e-8)
                    psnr_list.append(psnr.item())

                    # Masked PSNR
                    # Union of GT non-black pixels and Predicted Alpha mask
                    gt_mask = (gt_img.sum(dim=0, keepdim=True) > 1e-4).float()
                    pred_mask = (render_alpha > 1e-4).float()
                    eval_mask = torch.max(gt_mask, pred_mask)

                    if eval_mask.sum() > 0:
                        masked_mse = torch.sum(((gt_img - render_img) ** 2) * eval_mask) / (eval_mask.sum() * 3)
                        masked_psnr = -10 * torch.log10(masked_mse + 1e-8)
                    else:
                        masked_psnr = torch.tensor(0.0, device=gt_img.device)
                    masked_psnr_list.append(masked_psnr.item())

                    # LPIPS expects images in range [-1, 1], so we map [0, 1] -> [-1, 1]
                    gt_img_lpips = gt_img.unsqueeze(0) * 2.0 - 1.0
                    render_img_lpips = render_img.unsqueeze(0) * 2.0 - 1.0
                    lpips_val = loss_fn_vgg(render_img_lpips, gt_img_lpips)
                    lpips_list.append(lpips_val.item())

                    # SSIM
                    ssim_val = compute_ssim(render_img, gt_img)
                    ssim_list.append(ssim_val)

                    # CLIP Alignment
                    gt_img_clip = clip_normalize(F.interpolate(gt_img.unsqueeze(0).to(torch.float32), size=(224, 224), mode='bilinear', align_corners=False))
                    render_img_clip = clip_normalize(F.interpolate(render_img.unsqueeze(0).to(torch.float32), size=(224, 224), mode='bilinear', align_corners=False))
                    render_feat = clip_model(render_img_clip).image_embeds
                    gt_feat = clip_model(gt_img_clip).image_embeds
                    clip_val = F.cosine_similarity(render_feat, gt_feat, dim=-1).mean().item()
                    clip_list.append(clip_val)

                    results_meta["samples"].append({
                        "id": sample_name,
                        "psnr": psnr.item(),
                        "masked_psnr": masked_psnr.item(),
                        "ssim": ssim_val,
                        "lpips": lpips_val.item(),
                        "clip": clip_val,
                    })

                    # Video of spinning mesh for project page
                    if save_videos and (video_indices is None or (count + i) in video_indices):
                        from trellis2.utils import render_utils
                        # 360° orbit starting and ending at the condition camera's yaw/pitch.
                        _R_v = cond_extrinsics[i][:3, :3]
                        _t_v = cond_extrinsics[i][:3, 3]
                        _cam_pos_v = -(_R_v.T @ _t_v)
                        _r_v = _cam_pos_v.norm().item()
                        _pitch_v = math.asin(max(-1.0, min(1.0, (_cam_pos_v[2] / _r_v).item())))
                        _yaw_v = math.atan2(_cam_pos_v[0].item(), _cam_pos_v[1].item())
                        _nf = 240
                        _yaws_v = [_yaw_v - 2 * math.pi * f / _nf for f in range(_nf)]
                        _pitches_v = [_pitch_v] * _nf
                        _exts_v, _intrs_v = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(_yaws_v, _pitches_v, 2.0, 40.0)
                        video_frames = render_utils.render_frames(rep, _exts_v, _intrs_v, {'resolution': video_resolution, 'ssaa': 2}, envmap=current_envmap, verbose=False)
                        # Composite shaded over white using alpha
                        mp4_frames = []
                        for fi in range(len(video_frames['shaded'])):
                            sh = video_frames['shaded'][fi].astype(np.float32) / 255.0
                            al = video_frames['alpha'][fi].astype(np.float32) / 255.0
                            if al.ndim == 3:
                                al = al[..., :1]
                            mp4_frames.append(((sh * al + (1 - al)).clip(0, 1) * 255).astype(np.uint8))
                        imageio.mimsave(os.path.join(videos_dir, f"{sample_name}.mp4"), mp4_frames, fps=30, quality=7, macro_block_size=1)
                        del video_frames, mp4_frames

                    # --- Multi-view Generation for FID + per-sample plausibility metrics ---
                    sample_mv_psnrs, sample_mv_masked_psnrs, sample_mv_ssims, sample_mv_lpips, sample_mv_clips = [], [], [], [], []
                    for v in range(num_fid_views):
                        cam_seed = seed + (count + i) * 1000 + v
                        ext, intr = get_random_camera(cam_seed, device='cuda')

                        # Render Pred (PNG — lossless for FID reference integrity)
                        pred_res_fid = renderer.render(rep, ext, intr, envmap=current_envmap)
                        pred_alpha_mv = pred_res_fid['alpha'].clamp(0, 1)
                        if pred_alpha_mv.dim() == 2:
                            pred_alpha_mv = pred_alpha_mv.unsqueeze(0)
                        pred_img_tensor = pred_res_fid['shaded'].clamp(0, 1) * pred_alpha_mv
                        pred_img_np_fid = (pred_img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(pred_views_dir, f"{sample_name}_v{v:02d}.png"),
                                    cv2.cvtColor(pred_img_np_fid, cv2.COLOR_RGB2BGR))

                        # Render or load GT (cached as PNG — lossless)
                        gt_view_path = os.path.join(gt_fid_dir, f"{sample_name}_v{v:02d}.png")
                        if not os.path.exists(gt_view_path):
                            gt_res_fid = renderer.render(reps_gt[i], ext, intr, envmap=current_envmap)
                            gt_img_tensor_mv = (gt_res_fid['shaded'].clamp(0, 1) * gt_res_fid['alpha'].clamp(0, 1))
                            gt_img_np_fid = (gt_img_tensor_mv.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                            cv2.imwrite(gt_view_path, cv2.cvtColor(gt_img_np_fid, cv2.COLOR_RGB2BGR))
                        else:
                            gt_bgr = cv2.imread(gt_view_path, cv2.IMREAD_COLOR)
                            gt_img_np_fid = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
                            gt_img_tensor_mv = torch.from_numpy(gt_img_np_fid.astype(np.float32) / 255.0).permute(2, 0, 1).cuda()

                        # Multi-view metrics (plausibility proxy)
                        mv_mse = torch.mean((pred_img_tensor - gt_img_tensor_mv) ** 2)
                        sample_mv_psnrs.append((-10 * torch.log10(mv_mse + 1e-8)).item())

                        # Masked PSNR: union of pred alpha and gt foreground
                        gt_mask_mv = (gt_img_tensor_mv.sum(dim=0, keepdim=True) > 1e-4).float()
                        pred_mask_mv = (pred_alpha_mv > 1e-4).float()
                        eval_mask_mv = torch.max(gt_mask_mv, pred_mask_mv)
                        if eval_mask_mv.sum() > 0:
                            mv_masked_mse = torch.sum(((pred_img_tensor - gt_img_tensor_mv) ** 2) * eval_mask_mv) / (eval_mask_mv.sum() * 3)
                            mv_masked_psnr_val = -10 * torch.log10(mv_masked_mse + 1e-8)
                        else:
                            mv_masked_psnr_val = torch.tensor(0.0, device=pred_img_tensor.device)
                        sample_mv_masked_psnrs.append(mv_masked_psnr_val.item())
                        sample_mv_ssims.append(compute_ssim(pred_img_tensor, gt_img_tensor_mv))
                        mv_lpips_val = loss_fn_vgg(
                            pred_img_tensor.unsqueeze(0) * 2.0 - 1.0,
                            gt_img_tensor_mv.unsqueeze(0) * 2.0 - 1.0
                        )
                        sample_mv_lpips.append(mv_lpips_val.item())
                        pred_clip_mv = clip_normalize(F.interpolate(pred_img_tensor.unsqueeze(0).float(), size=(224, 224), mode='bilinear', align_corners=False))
                        gt_clip_mv = clip_normalize(F.interpolate(gt_img_tensor_mv.unsqueeze(0).float(), size=(224, 224), mode='bilinear', align_corners=False))
                        clip_mv = F.cosine_similarity(clip_model(pred_clip_mv).image_embeds, clip_model(gt_clip_mv).image_embeds, dim=-1).mean().item()
                        sample_mv_clips.append(clip_mv)

                    mv_psnr_list.append(float(np.mean(sample_mv_psnrs)))
                    mv_masked_psnr_list.append(float(np.mean(sample_mv_masked_psnrs)))
                    mv_ssim_list.append(float(np.mean(sample_mv_ssims)))
                    mv_lpips_list.append(float(np.mean(sample_mv_lpips)))
                    mv_clip_list.append(float(np.mean(sample_mv_clips)))
                    results_meta["samples"][-1]["mv_psnr"] = mv_psnr_list[-1]
                    results_meta["samples"][-1]["mv_masked_psnr"] = mv_masked_psnr_list[-1]
                    results_meta["samples"][-1]["mv_ssim"] = mv_ssim_list[-1]
                    results_meta["samples"][-1]["mv_lpips"] = mv_lpips_list[-1]
                    results_meta["samples"][-1]["mv_clip"] = mv_clip_list[-1]

                except RuntimeError as e:
                    print(f"Render failed for sample {count + i}: {e}")

            count += batch_size
            processed += batch_size
            pbar.update(batch_size)

            # Explicit cleanup to prevent memory overlapping between massive meshes
            if 'res' in locals():
                del res
            del pred_z, reps, noise, args, cond_update_fn
            if 'render_res' in locals():
                del render_res, render_img, render_alpha, gt_img, render_img_lpips, gt_img_lpips, mse, psnr, lpips_val, ssim_val, clip_val
                del eval_mask, gt_mask, pred_mask, masked_mse
                del pred_res_fid, pred_img_tensor, pred_img_np_fid, pred_alpha_mv
                del sample_mv_psnrs, sample_mv_masked_psnrs, sample_mv_ssims, sample_mv_lpips, sample_mv_clips
                del gt_mask_mv, pred_mask_mv, eval_mask_mv, mv_masked_psnr_val
                if 'mv_masked_mse' in locals():
                    del mv_masked_mse
                if 'gt_res_fid' in locals():
                    del gt_res_fid
                if 'gt_img_tensor_mv' in locals():
                    del gt_img_tensor_mv, gt_img_np_fid
            if 'current_envmap' in locals():
                del current_envmap
            if 'rep' in locals():
                del rep
            if 'reps_gt' in locals():
                del reps_gt
            del shape_z, cond_imgs, cond_extrinsics, cond_intrinsics
            del batch_data
            gc.collect()
            torch.cuda.empty_cache()

    avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    avg_masked_psnr = float(np.mean(masked_psnr_list)) if masked_psnr_list else 0.0
    avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    avg_lpips = float(np.mean(lpips_list)) if lpips_list else 0.0
    avg_clip = float(np.mean(clip_list)) if clip_list else 0.0
    avg_mv_psnr = float(np.mean(mv_psnr_list)) if mv_psnr_list else 0.0
    avg_mv_masked_psnr = float(np.mean(mv_masked_psnr_list)) if mv_masked_psnr_list else 0.0
    avg_mv_ssim = float(np.mean(mv_ssim_list)) if mv_ssim_list else 0.0
    avg_mv_lpips = float(np.mean(mv_lpips_list)) if mv_lpips_list else 0.0
    avg_mv_clip = float(np.mean(mv_clip_list)) if mv_clip_list else 0.0

    results_meta["summary"] = {
        "fidelity": {
            "mean_psnr": avg_psnr,
            "mean_masked_psnr": avg_masked_psnr,
            "mean_ssim": avg_ssim,
            "mean_lpips": avg_lpips,
            "mean_clip": avg_clip,
        },
        "plausibility": {
            "mean_mv_psnr": avg_mv_psnr,
            "mean_mv_masked_psnr": avg_mv_masked_psnr,
            "mean_mv_ssim": avg_mv_ssim,
            "mean_mv_lpips": avg_mv_lpips,
            "mean_mv_clip": avg_mv_clip,
        },
        "num_samples": count
    }

    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump(results_meta, f, indent=4)

    print(f"\nEvaluation complete over {count} samples")
    print(f"\n--- Fidelity (vs. conditioning view) ---")
    print(f"  PSNR:        {avg_psnr:.4f}")
    print(f"  Masked PSNR: {avg_masked_psnr:.4f}")
    print(f"  SSIM:        {avg_ssim:.4f}")
    print(f"  LPIPS:       {avg_lpips:.4f}")
    print(f"  CLIP:        {avg_clip:.4f}")
    print(f"\n--- Plausibility (avg over {num_fid_views} random views vs GT) ---")
    print(f"  MV-PSNR:        {avg_mv_psnr:.4f}")
    print(f"  MV-Masked PSNR: {avg_mv_masked_psnr:.4f}")
    print(f"  MV-SSIM:        {avg_mv_ssim:.4f}")
    print(f"  MV-LPIPS:       {avg_mv_lpips:.4f}")
    print(f"  MV-CLIP:        {avg_mv_clip:.4f}")

    print("\nCalculating Multi-view FID...")
    import subprocess
    try:
        fid_cmd = ["python", "-m", "pytorch_fid", gt_fid_dir, pred_views_dir, "--device", "cuda:0"]
        print(f"Executing: {' '.join(fid_cmd)}")

        result = subprocess.run(fid_cmd, capture_output=True, text=True, check=True)
        print(result.stdout)

        # Parse FID and save to metrics.json
        fid_score = None
        for line in result.stdout.split('\n'):
            if line.startswith("FID:"):
                fid_score = float(line.split("FID:")[-1].strip())

        if fid_score is not None:
            results_meta["summary"]["fid"] = fid_score
            with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
                json.dump(results_meta, f, indent=4)
        print(f"FID Evaluation Complete! GT views used from: {gt_fid_dir}")

    except subprocess.CalledProcessError as e:
        print(f"FID calculation failed!\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Experiment config file (defaults to load_dir/config.json)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--load_dir', type=str, default='', help='Load directory, default to output_dir')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint step to resume training, default to latest')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Data directory')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of examples to evaluate')
    parser.add_argument('--start_idx', type=int, default=0, help='Dataset index to start from (skip earlier samples)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic evaluation')
    parser.add_argument('--gt_fid_dir', type=str, default='eval_gt_fid_views', help='Directory to cache GT multi-views for FID')
    parser.add_argument('--grad_cfg_strength', type=float, default=1.0, help='Gradient CFG strength for with_grad/with_prox_grad modes (1.0 = off, >1.0 amplifies gradient influence using cached unguided prediction)')
    parser.add_argument('--save_videos', action='store_true', help='Save a spinning-mesh MP4 for each sample into videos/')
    parser.add_argument('--video_resolution', type=int, default=1024, help='Resolution of saved videos (default 1024)')
    parser.add_argument('--video_indices', type=str, default=None, help='Comma-separated 0-based sample indices for which to save a video (requires --save_videos), e.g. "0,3,7"')
    parser.add_argument('--num_steps', type=int, default=12, help='Number of ODE sampling steps (default 12)')
    parser.add_argument('--guidance_strength', type=float, default=1.0, help='Classifier-free guidance strength (default 1.0 = no CFG amplification)')
    parser.add_argument('--t_thresh', type=float, default=0.0, help='Prior-step feedback approximation threshold (with_grad/with_prox_grad only): for t < t_thresh, reuse the previous step\'s cached x_hat_0 instead of a fresh look-ahead pass. Uses the codebase\'s t convention (t=1 noise -> t=0 clean, opposite of the paper\'s FM convention). 0.0 (default) = full two-pass sampling; 1.0 = shortcut on every step after the first. Not compatible with --grad_cfg_strength != 1.0.')

    opt = parser.parse_args()
    opt.load_dir = opt.load_dir if opt.load_dir != '' else opt.output_dir

    if opt.config is None:
        opt.config = os.path.join(opt.load_dir, 'config.json')
        if not os.path.exists(opt.config):
            raise FileNotFoundError(f"Config file not found at {opt.config}. Please provide it via --config.")

    config = json.load(open(opt.config, 'r'))
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)

    cfg = find_ckpt(cfg)
    _video_indices = set(int(x) for x in opt.video_indices.split(',') if x.strip()) if opt.video_indices else None
    evaluate(cfg, num_samples=opt.num_samples, seed=opt.seed, gt_fid_dir=opt.gt_fid_dir, grad_cfg_strength=opt.grad_cfg_strength, save_videos=opt.save_videos, video_resolution=opt.video_resolution, guidance_strength=opt.guidance_strength, start_idx=opt.start_idx, video_indices=_video_indices, t_thresh=opt.t_thresh)
