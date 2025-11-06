# activate Singularity
singularity exec \
  --overlay /scratch/rc5832/vggt/overlay-50G-10M.ext3 \
  --overlay /vast/xl3136/DATA/wanderland_eval.sqf:ro \
  /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash

# run code
cd /scratch/rc5832/vggt/code

run-vggt.sbatch

run-eval.sbatch

or sbatch run_vggt_plus_eval.sbatch

python collect_pose_eval_results.py --root /scratch/rc5832/vggt_results

