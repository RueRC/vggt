import random
import glob
import os
import copy
import torch
import torch.nn.functional as F
import numpy as np

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import pycolmap

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
# ===== depth/points no imports =====
# from vggt.utils.geometry import unproject_depth_map_to_point_map
# from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track


# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo (cams only; depth disabled)")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction (ignored)")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )

    # parser.add_argument(
    #     "--max_num_img", type=int, default=None,
    #     help="Uniformly subsample images to this count; always keep the first image."
    # )

    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for COLMAP files")

    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518):
    # images: [S, 3, H, W]  （注：下方会加 batch 维，保持你原版 aggregator 的期望形状）

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension -> [1, S, 3, 518, 518]
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras（only）
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    extrinsic = extrinsic.squeeze(0).cpu().numpy()  # [S, 4, 4]
    intrinsic = intrinsic.squeeze(0).cpu().numpy()  # [S, 3, 3]
    return extrinsic, intrinsic


def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))
    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera estimation (depth disabled)
    model = VGGT()

    # FROM LOCAL
    # ckpt = torch.load("/media/huge/Huge/lab/vggt/pretrained_model/model.pt", map_location="cpu")
    # model.load_state_dict(ckpt)

    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")

    image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")

    # # ---- Uniform Down Sampling ----
    N = len(image_path_list)

    if 100 <= N <= 1000: # TODO 500 <= N <= 1000
        # Downsample by factor 2 → target ~ N/2 images
        K = max(1, N // 20) # TODO 2
        sel_idx = np.linspace(0, N - 1, num=K, dtype=int).tolist()
        print(f"[Sampling Strategy] {N} images → downsample 1/2 → {K} images")

    elif 1001 <= N <= 1500:
        # Downsample by factor 3 → target ~ N/3 images
        K = max(1, N // 30) # TODO 3
        sel_idx = np.linspace(0, N - 1, num=K, dtype=int).tolist()
        print(f"[Sampling Strategy] {N} images → downsample 1/3 → {K} images")

    elif N > 1500:
        # Recompute uniform interval to get **exactly 500 images**
        K = 50 # TODO 500
        sel_idx = np.linspace(0, N - 1, num=K, dtype=int).tolist()
        print(f"[Sampling Strategy] {N} images → uniform sample to 500 images")

    else:
        # No downsampling
        sel_idx = list(range(N))
        print(f"[Sampling Strategy] {N} images → use all")

    # Apply selection
    image_path_list = [image_path_list[i] for i in sel_idx]
    base_image_path_list = [os.path.basename(p) for p in image_path_list]
    print(f"[Sampling] final count: {len(image_path_list)} (from original {N})")

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # ---- Cameras only: run without depth ----
    extrinsic, intrinsic = run_VGGT(model, images, dtype, vggt_fixed_resolution)

    # ---- Build empty-points reconstruction via your helper (cam_id 等交给封装) ----
    shared_camera = args.shared_camera
    camera_type = args.camera_type
    image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution], dtype=np.int32)

    # empty points3D.bin file
    points_3d = np.empty((0, 3), dtype=np.float32)
    points_xyf = np.empty((0, 3), dtype=np.float32)
    points_rgb = np.empty((0, 3), dtype=np.uint8)

    print("Converting to COLMAP format (cameras & images only; no points3D).")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )

    reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    # print(f"Saving reconstruction to {args.scene_dir}/sparse_vggt")
    # sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse_vggt")
    # os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    # reconstruction.write(sparse_reconstruction_dir)
    sparse_reconstruction_dir = args.out_dir if args.out_dir else os.path.join(args.scene_dir, "sparse_vggt")
    print(f"Saving reconstruction to {sparse_reconstruction_dir}")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # for points3D
    # p3d_path = os.path.join(sparse_reconstruction_dir, "points3D.bin")
    # if os.path.exists(p3d_path):
    #     os.remove(p3d_path)

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = int(real_image_size[0])
            pycamera.height = int(real_image_size[1])

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

This variant disables depth/points entirely to be compatible with
aggregator's "keep only last outputs" memory-saving tweak.

Output (unchanged layout):
    sparse_vggt/
      - cameras.bin
      - images.bin
      - points3D.bin  (empty; optionally removed in code)
"""
