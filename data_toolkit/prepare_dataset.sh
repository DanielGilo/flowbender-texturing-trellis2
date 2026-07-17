#!/bin/bash
# Runs the full data preprocessing pipeline for a single dataset subset, from raw metadata
# through to a train/test split ready for training/evaluation.
#
# Usage: bash data_toolkit/prepare_dataset.sh <SUBSET> <ROOT> [SOURCE]
#   SUBSET: ObjaverseXL | Toys4k
#   ROOT:   output directory for the dataset (e.g. datasets/ObjaverseXL_sketchfab)
#   SOURCE: only used for ObjaverseXL (sketchfab|github), default sketchfab
#   MAX_WORKERS (env var, optional): worker count for the CPU-bound stages (mesh/PBR dump,
#     O-Voxel conversion). Defaults to $(nproc). Override with e.g. MAX_WORKERS=8 bash ...
#
# Prerequisites:
#   - ObjaverseXL: none, download is fully automated.
#   - Toys4k: manually download toys4k_blend_files.zip (see data_toolkit/README.md /
#     https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k) and place it at
#     <ROOT>/raw/toys4k_blend_files.zip before running this script.

set -e

SUBSET=$1
ROOT=$2
SOURCE=${3:-sketchfab}

# CPU-bound stages (Blender dump, O-Voxel conversion) parallelize across this many workers.
# dual_grid.py/voxelize_pbr.py default to only 4 workers if not set explicitly, regardless of
# core count, so we pass $(nproc) here explicitly.
MAX_WORKERS=${MAX_WORKERS:-$(nproc)}

if [ -z "$SUBSET" ] || [ -z "$ROOT" ]; then
    echo "Usage: bash data_toolkit/prepare_dataset.sh <SUBSET> <ROOT> [SOURCE]"
    exit 1
fi

SOURCE_ARG=""
if [ "$SUBSET" == "ObjaverseXL" ]; then
    SOURCE_ARG="--source $SOURCE"
fi

echo "=== [1/6] Metadata + download ==="
python data_toolkit/build_metadata.py "$SUBSET" --root "$ROOT" $SOURCE_ARG
python data_toolkit/download.py "$SUBSET" --root "$ROOT"
python data_toolkit/build_metadata.py "$SUBSET" --root "$ROOT"

echo "=== [2/6] Mesh + PBR dump ==="
python data_toolkit/dump_mesh.py "$SUBSET" --root "$ROOT" --max_workers "$MAX_WORKERS"
python data_toolkit/dump_pbr.py "$SUBSET" --root "$ROOT" --max_workers "$MAX_WORKERS"
python data_toolkit/build_metadata.py "$SUBSET" --root "$ROOT"

echo "=== [3/6] O-Voxel conversion (resolution 512) ==="
python data_toolkit/dual_grid.py "$SUBSET" --root "$ROOT" --resolution 512 --max_workers "$MAX_WORKERS"
python data_toolkit/voxelize_pbr.py "$SUBSET" --root "$ROOT" --resolution 512 --max_workers "$MAX_WORKERS"
python data_toolkit/build_metadata.py "$SUBSET" --root "$ROOT"

echo "=== [4/6] Encode shape + PBR latents ==="
python data_toolkit/encode_shape_latent.py --root "$ROOT" --resolution 512
python data_toolkit/encode_pbr_latent.py --root "$ROOT" --resolution 512
python data_toolkit/build_metadata.py "$SUBSET" --root "$ROOT"

echo "=== [5/6] Render conditioning views (GPU-native PBR renderer) ==="
python data_toolkit/render_cond_pbr_mesh_renderer.py --root "$ROOT" --num_cond_views 16
# Blender-based alternative (unused in this framework, kept for reference):
# python data_toolkit/render_cond.py "$SUBSET" --root "$ROOT" --num_cond_views 16
python data_toolkit/build_metadata.py "$SUBSET" --root "$ROOT"

echo "=== [6/6] Train/test split ==="
python data_toolkit/split_dataset.py --root "$ROOT" --test_ratio 0.1

echo "Done. Dataset ready at $ROOT"
