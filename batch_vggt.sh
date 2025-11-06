#!/usr/bin/env bash
set -euo pipefail

ROOT="/wanderland_eval"
LOGDIR="/scratch/rc5832/logs"
PY="/ext3/miniconda3/envs/vggt/bin/python"
SCRIPT="/scratch/rc5832/vggt/code/demo_colmap.py"

## --max_num_img
##MAX_NUM_IMG="${MAX_NUM_IMG:-}"
#MAX_NUM_IMG=160

# CPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$LOGDIR"

find "$ROOT" -type d -name images -print0 | while IFS= read -r -d '' imgdir; do
  scene_dir="$(dirname "$imgdir")"
  out="/scratch/rc5832/vggt/sparse_vggt"

  if [[ -s "${out}/images.bin" ]]; then
    echo "[SKIP] $scene_dir (already has sparse_vggt)"
    continue
  fi

  echo "[RUN ] $scene_dir"
  log="${LOGDIR}/$(echo "$scene_dir" | sed 's#/#_#g').log"

  ARGS=( "--scene_dir=$scene_dir" )
#  if [[ -n "${MAX_NUM_IMG}" ]]; then
#    ARGS+=( "--max_num_img=${MAX_NUM_IMG}" )
#  fi

  # log: failed.txt
  if "$PY" "$SCRIPT" "${ARGS[@]}" 2>&1 | tee "$log"; then
    echo "[DONE] $scene_dir"
  else
    echo "$scene_dir" >> "${LOGDIR}/failed.txt"
    echo "[FAIL] $scene_dir (logged to $log)"
  fi
done
