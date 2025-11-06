#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/huge/Game/test" #/media/huge/Game/squashfs-root/wanderland_eval
PY="/home/huge/anaconda3/envs/vggt/bin/python"
EVAL_SCRIPT="/media/huge/Huge/lab/vggt/benchmark/reconstruction/eval_colmap_poses_safe.py"
MATCH_MODE="exact"

LOGDIR="${ROOT}/_eval_logs"
mkdir -p "$LOGDIR"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

: > "${LOGDIR}/failed_eval.txt"

find "$ROOT" -mindepth 1 -maxdepth 1 -type d | sort | while read -r scene_dir; do
  base="$(basename "$scene_dir")"

  pred_dir="$scene_dir/sparse_vggt"
  gt_dir="$scene_dir/sparse/0"
  out_txt="$scene_dir/${base}.txt"
  log="${LOGDIR}/${base}.log"

  # check
  if [[ ! -s "$pred_dir/images.bin" ]]; then
    echo "[SKIP] $base (missing sparse_vggt/images.bin)"
    continue
  fi
  if [[ ! -s "$gt_dir/images.bin" ]]; then
    echo "[SKIP] $base (missing sparse/0/images.bin)"
    continue
  fi

  echo "[EVAL] $base → logging to $log"
  if "$PY" "$EVAL_SCRIPT" "$pred_dir" "$gt_dir" --match "$MATCH_MODE" --output "$out_txt" \
      2>&1 | tee "$log"; then
    echo "[DONE] $base  (result: $out_txt)"
  else
    echo "$scene_dir" >> "${LOGDIR}/failed_eval.txt"
    echo "[FAIL] $base (logged to $log)"
  fi

done