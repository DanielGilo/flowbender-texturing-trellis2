# FlowBender: 3D Mesh Texturing (TRELLIS.2)

This repository is the 3D mesh texturing implementation for **FlowBender**, part of the main
FlowBender repository. It is a fork of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
(Xiang et al., 2025), extending its Stage-3 texture flow model with FlowBender's closed-loop,
feedback-conditioned training. See the main FlowBender repository for the full method and its
other tasks (image-to-image translation, restoration).

**Task scope**: this repo focuses on PBR texture generation conditioned on a 3D geometry latent and a
single conditioning image — it does not generate geometry itself. The forward operator is a
differentiable PBR renderer that re-renders the predicted texture from the conditioning viewpoint.

It implements two FlowBender training modes for texturing — gradient feedback w.r.t. the noisy
latent (`with_grad`) and w.r.t. the denoised prediction (`with_prox_grad`, the strongest variant
reported in the paper) — alongside `vanilla` (Standard-FT) and `vanillapp` (FT + L_align) supervised
baselines.

**Relative to upstream TRELLIS.2**, this fork modifies:
- **Training**: adds LoRA fine-tuning and the feedback-aware training loop (`trellis2/trainers/flow_matching/sparse_flow_matching.py`) implementing the four modes above.
- **Data processing**: in contrast to upstream TRELLIS.2, which uses Blender to render the conditioning views for training, we use the differentiable PBR renderer both for data generation and as the forward operator during training and evaluation, keeping the two consistent. Dataset loaders additionally record camera extrinsics/intrinsics, needed to re-render the same viewpoint during training. We provide data processing for the two datasets used in this paper (ObjaverseXL, Toys4k).
- **Evaluation**: `evaluate.py` computes the fidelity/plausibility metrics reported in the paper (PSNR, masked-PSNR, SSIM, LPIPS, CLIP similarity, multi-view FID).

The rest of this README covers environment setup, data preprocessing, running training for each
mode, and reproducing the evaluation numbers.

## Setup and Installation


### Installation Steps

This repo is a submodule of the main FlowBender repository, cloned there with `o-voxel` already
initialized. Working standalone instead? Run `git submodule update --init --recursive` first.

1. Install dependencies (creates a `trellis2` conda env; drop `--new-env` to reuse an existing one):
    ```sh
    . ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
    ```
    See `. ./setup.sh --help` for flag details. No `flash-attn` support (e.g. V100)? Install
    `xformers` and set `ATTN_BACKEND=xformers` instead.

2. Install the extra dependencies FlowBender's training/eval code needs (not yet in `setup.sh`):
    ```sh
    pip install peft pytorch-fid scikit-image
    ```

3. Install data preprocessing dependencies (see [Data Preprocessing](#data-preprocessing)):
    ```sh
    . ./data_toolkit/setup.sh
    ```

## Pretrained Weights

Fine-tuning starts from the pretrained **TRELLIS.2-4B** Stage-3 texture model, fetched
automatically from [Hugging Face](https://huggingface.co/microsoft/TRELLIS.2-4B). You'll need to be authenticated
with Hugging Face (`huggingface-cli login` or an `HF_TOKEN`) — tokens can expire after a few days,
in which case you'll need to re-authenticate.

## Data Preprocessing

Training uses **ObjaverseXL** (sketchfab subset, ~8K assets); evaluation additionally uses
**Toys4k** (held-out). Both are indexed via metadata from the
[TRELLIS-500K](https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K) dataset, but raw asset
acquisition differs:

- **ObjaverseXL**: fully automated, no manual step.
- **Toys4k**: fill out the [form](https://forms.gle/w7Zf82umwaKxr9L7A) on the
  [Toys4k page](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k), download
  `toys4k_blend_files.zip`, and place it at `<ROOT>/raw/toys4k_blend_files.zip`.

Everything else — metadata, download, mesh/PBR dump, O-Voxel conversion, shape/PBR latent
encoding, conditioning-view rendering, and train/test split — runs via a single script:

```sh
bash data_toolkit/prepare_dataset.sh <SUBSET> <ROOT>   # SUBSET: ObjaverseXL | Toys4k
```

`<ROOT>` (e.g. `datasets/ObjaverseXL_sketchfab`, `datasets/Toys4k`) is what you'll point
`--data_dir` at during training/eval. See `data_toolkit/prepare_dataset.sh` for the individual
stages if you need to re-run or debug one of them.

This can take some time — the mesh/PBR dump and O-Voxel
conversion stages are CPU-bound (parallelized across `MAX_WORKERS`, default `$(nproc)`), while
latent encoding and conditioning-view rendering are GPU-bound. A machine with both a capable GPU
and many CPU cores is recommended; `MAX_WORKERS=<N> bash data_toolkit/prepare_dataset.sh ...`
overrides the worker count for the CPU-bound stages.

To sanity-check the pipeline (or debug a single stage) on a handful of assets instead of the
full dataset, every stage script accepts `--instances <sha256_a,sha256_b,...>` (or a path to a
newline-separated file of sha256s) to restrict processing to those assets only.

## Running Training

Four training modes are reported in the paper, each with its own config under `configs/gen/`:

| Config | Mode | Description |
|---|---|---|
| `tex_vanilla.json` | `vanilla` | Standard supervised fine-tuning baseline |
| `tex_vanillapp.json` | `vanillapp` | FT + L_align, a render-consistency alignment loss baseline from the paper |
| `tex_with_grad.json` | `with_grad` | FlowBender, gradient feedback w.r.t. the noisy latent |
| `tex_with_prox_grad.json` | `with_prox_grad` | FlowBender, gradient feedback w.r.t. the denoised prediction (strongest variant) |

All four fine-tune the pretrained TRELLIS.2-4B Stage-3 texture model via LoRA. `with_grad`/
`with_prox_grad` feed a gradient signal to the denoiser as an extra input; `vanillapp` instead adds
an additional render-consistency loss term (no extra denoiser input); `vanilla` is plain
supervised fine-tuning.

### Command

```sh
python train.py \
  --config configs/gen/tex_with_prox_grad.json \
  --output_dir results/tex_with_prox_grad \
  --data_dir '{"ObjaverseXL_sketchfab": {
      "base": "datasets/ObjaverseXL_sketchfab",
      "shape_latent": "datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_512",
      "pbr_latent": "datasets/ObjaverseXL_sketchfab/pbr_latents/tex_enc_next_dc_f16c32_fp16_512",
      "render_cond": "datasets/ObjaverseXL_sketchfab/renders_cond"
  }}'
```

The dataset paths match the output of [Data Preprocessing](#data-preprocessing) (`<ROOT>` and its
subdirectories). Swap `--config` for any of the four configs above to train the other modes.

### Useful flags
- `--num_gpus <N>`: GPUs per node (default: all available).
- `--ckpt latest`: resume from the latest checkpoint in `--output_dir` (default behavior).
- `--tryrun`: dry run — builds the model/dataset/trainer without training, useful for sanity-checking a config.
- `--num_nodes`, `--node_rank`, `--master_addr`, `--master_port`: multi-node training.

### Changing hyperparameters
All hyperparameters live in the config JSON — edit it directly, no code changes needed. Commonly
tuned ones:
- `models.denoiser.args.lora_rank` / `lora_alpha` / `lora_dropout` — LoRA capacity.
- `trainer.args.optimizer.args.lr` (and `weight_decay`, `betas`) — learning rate.
- `trainer.args.p_uncond` / `p_unguided` — classifier-free-guidance dropout / unguided-iteration probability (`with_grad`, `with_prox_grad` only).
- `trainer.args.render_loss_weight` — alignment-loss weight (`vanillapp` only).
- `trainer.args.max_steps`, `batch_size_per_gpu`, `batch_split` — training length / batch size.

## Evaluation

```sh
python evaluate.py \
  --output_dir results/tex_with_prox_grad \
  --data_dir '{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", ...}}' \
  --num_samples 100
```
`--output_dir` should be the same directory used as `--output_dir` during training (checkpoint and
`config.json` are loaded from there, unless `--load_dir`/`--config` are set separately).

Reports **fidelity** (PSNR, masked-PSNR, SSIM, LPIPS, CLIP similarity vs. the conditioning view)
and **plausibility** (the same metrics averaged over 50 random views, plus multi-view FID against
cached GT renders), written to `metrics.json` in the run's output directory.

### Flags matching paper terminology
- `--grad_cfg_strength w`: the paper's **Optional CFG** (`with_grad`/`with_prox_grad` only) —
  blends the refined velocity v_ref with the cached look-ahead velocity v_LA at zero extra cost:
  `v_cfg = w · v_ref + (1-w) · v_LA`. `w=1.0` (default) disables it.
- `--t_thresh τ`: the paper's **prior-step feedback approximation** (`with_grad`/`with_prox_grad`
  only) — for sampling steps with `t < τ`, the feedback gradient is derived from the previous
  step's cached x̂₀ instead of a fresh look-ahead pass, cutting those steps from 2 denoiser
  evaluations to 1. **Note**: `t` here follows this codebase's convention (`t=1` is pure noise
  at the start of sampling, `t=0` is the clean sample at the end) — the *opposite* direction
  from the paper's FM notation (`t=0` to `t=1`). `τ=0.0` (default) always takes the full
  two-pass step, since no step ever reaches `t < 0`; `τ=1.0` uses the shortcut for every step
  after the first (which always runs the full pass, having no cached x̂₀ yet). Not compatible
  with `--grad_cfg_strength != 1.0` (no cached look-ahead velocity to blend with on shortcut
  steps).
- `--num_steps`: number of Euler sampling steps N (default 12).

Other flags: `--ckpt` (checkpoint step to load, default `latest`); `--save_videos` (off by
default — saves a spinning-mesh MP4 per sample, optionally narrowed to specific samples with
e.g. `--video_indices "0,3,7"`). Run `python evaluate.py --help` for the full list.

### Notes
- `--gt_fid_dir` (default `eval_gt_fid_views`) caches GT multi-view renders — reuse the same path
  across runs on the same dataset to avoid re-rendering GT views every time.
