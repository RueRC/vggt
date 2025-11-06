#python eval_colmap_poses_safe.py   /media/huge/Huge/lab/vggt/scene_data/scene0/colmap_output   /media/huge/Huge/lab/vggt/gt_data/scene0/sparse/0   --match exact   --output /media/huge/Huge/lab/vggt/scene_data/scene0/pose_eval_results.txt

#!/usr/bin/env python3
import os, sys, argparse, struct, numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

@dataclass
class ImgPose:
    image_id: int
    name: str
    camera_id: int
    qvec_wxyz: np.ndarray  # [w,x,y,z]
    tvec_w2c: np.ndarray   # world -> cam

def _safe_read_images_bin(path_to_model_file):
    """
    只读 images.bin 的 header（id, qvec, tvec, camera_id, name）并安全跳过 POINTS2D。
    约定（与当前写 bin 的脚本一致）：
      - name 为不定长字节，后跟 '\x00' 终结（不写长度）
      - num_points2D 为 uint64
      - 每个点为 (double x, double y, int64 point3D_id)  共 24 字节
    返回：name -> {'qvec_wxyz': (4,), 'tvec': (3,)} 的 dict
    """
    poses = {}
    with open(path_to_model_file, "rb") as f:
        # uint64: number of images
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            # int32 image_id
            image_id = struct.unpack("<i", f.read(4))[0]
            # 4*double q
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(8*4))
            # 3*double t
            tx, ty, tz = struct.unpack("<ddd", f.read(8*3))
            # int32 camera_id
            camera_id = struct.unpack("<i", f.read(4))[0]

            # name: 以 '\x00' 结尾的字节串
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"":  # EOF
                    raise EOFError("Unexpected EOF while reading image name")
                if c == b"\x00":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8", errors="strict")

            # uint64: num_points2D
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            # sanity：防止坏数据导致巨量 seek
            if num_points2D > 50_000_000:
                raise ValueError(f"num_points2D too large ({num_points2D}) at image {image_id}")

            # 跳过 2D 点：2*double + int64 (signed) = 24 bytes/pt
            to_skip = 24 * int(num_points2D)
            # chunked seek 防止极大 offset 一次性 seek 失败
            CHUNK = 1 << 30
            while to_skip > 0:
                step = CHUNK if to_skip > CHUNK else to_skip
                f.seek(step, 1)
                to_skip -= step

            q = np.array([qw, qx, qy, qz], dtype=np.float64)
            t = np.array([tx, ty, tz], dtype=np.float64)
            poses[name] = {"qvec_wxyz": q, "tvec": t}

    return poses




def _read_images_txt(path_txt: str) -> Dict[int, ImgPose]:
    poses = {}
    if not os.path.exists(path_txt): return poses
    with open(path_txt, "r") as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 10: continue
            iid = int(parts[0])
            q = np.array(list(map(float, parts[1:5])), dtype=np.float64)
            t = np.array(list(map(float, parts[5:8])), dtype=np.float64)
            cam_id = int(parts[8])
            name = parts[9]
            poses[iid] = ImgPose(iid, name, cam_id, q, t)
            _ = f.readline()  # 下一行是 2D 点，跳过
    return poses

def load_poses(model_dir: str):
    """
    读取 COLMAP sparse 目录的位姿，只用 images.{bin,txt}。
    优先用 pycolmap（若可用且能成功），否则走我们上面的安全 reader。
    返回：name -> {'qvec_wxyz': (4,), 'tvec': (3,)} 的 dict
    """
    import os
    images_bin = os.path.join(model_dir, "images.bin")
    images_txt = os.path.join(model_dir, "images.txt")

    # 先试 pycolmap
    try:
        import pycolmap
        rec = pycolmap.Reconstruction(model_dir)
        poses = {}
        for img_id, img in rec.images.items():
            # pycolmap 给的是（w,x,y,z）
            q = np.array([img.qvec[0], img.qvec[1], img.qvec[2], img.qvec[3]], dtype=np.float64)
            t = np.array([img.tvec[0], img.tvec[1], img.tvec[2]], dtype=np.float64)
            poses[img.name] = {"qvec_wxyz": q, "tvec": t}
        if len(poses) > 0:
            return poses
    except Exception as e:
        print(f"[info] pycolmap failed on {model_dir}: {e}")

    # 走安全 reader
    if os.path.exists(images_bin):
        return _safe_read_images_bin(images_bin)

    # 兜底：TXT（如果有）
    if os.path.exists(images_txt):
        poses = {}
        with open(images_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                elems = line.split()
                if len(elems) < 10:
                    continue
                # IMAGE_ID qw qx qy qz tx ty tz CAMERA_ID NAME
                qw, qx, qy, qz = map(float, elems[1:5])
                tx, ty, tz     = map(float, elems[5:8])
                name           = elems[9]
                poses[name] = {"qvec_wxyz": np.array([qw,qx,qy,qz], dtype=np.float64),
                               "tvec":      np.array([tx,ty,tz], dtype=np.float64)}
        return poses

    raise FileNotFoundError(f"No images.bin or images.txt under {model_dir}")



from scipy.spatial.transform import Rotation as R
import numpy as np

def imgpose_to_c2w(p: dict, name: str = "<unknown>"):
    """
    输入 p 是从 images.{bin,txt} 解析出的单张图位姿字典。
    兼容键名:
      - qvec_wxyz 或 qvec   (四元数顺序 [w,x,y,z])
      - tvec                 (平移向量)
    返回: 4x4 的 camera-to-world 矩阵 (c2w)
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    # ---- 取字段（兼容不同键名）----
    if "qvec_wxyz" in p:
        q = p["qvec_wxyz"]
    elif "qvec" in p:
        q = p["qvec"]
    else:
        raise KeyError(f"{name}: pose dict missing 'qvec_wxyz' / 'qvec'")

    if "tvec" not in p:
        raise KeyError(f"{name}: pose dict missing 'tvec'")

    # 转成 ndarray，确保形状正确
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    t = np.asarray(p["tvec"], dtype=np.float64).reshape(-1)

    if q.size != 4 or t.size != 3:
        raise ValueError(f"{name}: wrong shapes, q.shape={q.shape}, t.shape={t.shape}")

    # 数值检查
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(t))):
        raise ValueError(f"{name}: non-finite entries in q/t. q={q}, t={t}")

    # 归一化（避免“零范数”误判），并打印调试信息
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n == 0:
        # 打印出来看看是哪一张
        print(f"[bad quat] {name}: q={q} (norm={n}) -> fallback to identity")
        # 兜底（基本不会走到这）：用单位旋转
        R_w2c = np.eye(3)
    else:
        q = q / n
        # COLMAP 存的是 w2c，四元数顺序是 w,x,y,z
        # scipy 需要 x,y,z,w
        q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
        try:
            R_w2c = R.from_quat(q_xyzw).as_matrix()
        except Exception as e:
            # 万一四元数格式仍有异常，打印出来定位
            print(f"[quat->R fail] {name}: q_wxyz={q}, q_xyzw={q_xyzw}, err={e}")
            raise

    # 按 COLMAP 约定：X_c = R * X_w + t   （w2c）
    T_w2c = np.eye(4, dtype=np.float64)
    T_w2c[:3, :3] = R_w2c
    T_w2c[:3, 3]  = t

    # 取逆得到 c2w
    try:
        T_c2w = np.linalg.inv(T_w2c)
    except np.linalg.LinAlgError as e:
        print(f"[inv fail] {name}: T_w2c=\n{T_w2c}\nerr={e}")
        raise

    return T_c2w


def se3_align(pred_Ts: List[np.ndarray], gt_Ts: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    P = np.stack([T[:3,3] for T in pred_Ts]); G = np.stack([T[:3,3] for T in gt_Ts])
    Pc, Gc = P.mean(0), G.mean(0)
    P0, G0 = P - Pc, G - Gc
    H = P0.T @ G0
    U, _, Vt = np.linalg.svd(H)
    Rg = Vt.T @ U.T
    if np.linalg.det(Rg) < 0: Vt[-1] *= -1; Rg = Vt.T @ U.T
    t = Gc - Rg @ Pc
    return Rg, t

def sim3_align(pred_Ts: List[np.ndarray], gt_Ts: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float]:
    P = np.stack([T[:3,3] for T in pred_Ts]); G = np.stack([T[:3,3] for T in gt_Ts])
    Pc, Gc = P.mean(0), G.mean(0)
    P0, G0 = P - Pc, G - Gc
    s = (np.linalg.norm(G0, axis=1) / np.maximum(np.linalg.norm(P0, axis=1), 1e-12)).mean()
    H = (P0*s).T @ G0
    U, _, Vt = np.linalg.svd(H)
    Rg = Vt.T @ U.T
    if np.linalg.det(Rg) < 0: Vt[-1] *= -1; Rg = Vt.T @ U.T
    t = Gc - Rg @ (Pc * s)
    return Rg, t, s

def ate_trans(pred_Ts, gt_Ts):
    P = np.stack([T[:3,3] for T in pred_Ts]); G = np.stack([T[:3,3] for T in gt_Ts])
    return float(np.sqrt(np.mean(np.sum((P-G)**2, axis=1))))

def ate_rot(pred_Ts, gt_Ts):
    errs = []
    for Tp, Tg in zip(pred_Ts, gt_Ts):
        Rrel = Tg[:3,:3].T @ Tp[:3,:3]
        ang = np.arccos(np.clip((np.trace(Rrel)-1)/2, -1, 1))
        errs.append(np.degrees(ang))
    return float(np.sqrt(np.mean(np.array(errs)**2)))

def rte_trans_allpairs(pred_Ts, gt_Ts):
    P = np.stack([T[:3,3] for T in pred_Ts]); G = np.stack([T[:3,3] for T in gt_Ts])
    Pd = P[:,None,:]-P[None,:,:]; Gd = G[:,None,:]-G[None,:,:]
    E = np.linalg.norm(Pd-Gd, axis=2)
    iu = np.triu_indices(len(P), 1)
    vals = E[iu]
    return float(np.sqrt(np.mean(vals**2)))

def rte_trans_angle_allpairs(pred_Ts, gt_Ts):
    P = np.stack([T[:3,3] for T in pred_Ts]); G = np.stack([T[:3,3] for T in gt_Ts])
    Pd = P[:,None,:]-P[None,:,:]; Gd = G[:,None,:]-G[None,:,:]
    Pn = np.linalg.norm(Pd, axis=2); Gn = np.linalg.norm(Gd, axis=2)
    mask = (Pn>1e-8)&(Gn>1e-8)
    Pu = np.zeros_like(Pd); Gu = np.zeros_like(Gd)
    Pu[mask] = Pd[mask]/Pn[mask,None]; Gu[mask] = Gd[mask]/Gn[mask,None]
    dot = np.clip(np.sum(Pu*Gu, axis=2), -1, 1)
    ang = np.degrees(np.arccos(dot))
    iu = np.triu_indices(len(P), 1)
    valid = mask[iu]
    if not np.any(valid): return 0.0, np.array([])
    vals = ang[iu][valid]
    return float(np.sqrt(np.mean(vals**2))), vals

def rte_rot_allpairs(pred_Ts, gt_Ts):
    Rs = np.stack([T[:3,:3] for T in pred_Ts]); Gs = np.stack([T[:3,:3] for T in gt_Ts])
    n = len(Rs)
    angs = []
    for i in range(n):
        for j in range(i+1, n):
            Rpr = Rs[i] @ Rs[j].T
            Gpr = Gs[i] @ Gs[j].T
            Rrel = Gpr.T @ Rpr
            ang = np.degrees(np.arccos(np.clip((np.trace(Rrel)-1)/2, -1, 1)))
            angs.append(ang)
    angs = np.array(angs)
    return float(np.sqrt(np.mean(angs**2))), angs

def auc30(r_err_deg: np.ndarray, t_err_deg: np.ndarray, max_th=30):
    if len(r_err_deg)==0 or len(t_err_deg)==0: return 0.0
    m = np.maximum(r_err_deg, t_err_deg)
    bins = np.arange(max_th+1)
    hist, _ = np.histogram(m, bins=bins)
    hist = hist.astype(float) / max(1, len(m))
    return float(np.mean(np.cumsum(hist)))

def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("gt_dir")
    ap.add_argument("--output", "-o", default="pose_evaluation_results.txt")
    ap.add_argument("--match", choices=["exact","basename"], default="exact",
                    help="exact: 完全相同文件名；basename: 只比对去掉路径后的文件名")
    args = ap.parse_args()

    print("Reading prediction poses...")
    pred = load_poses(args.pred_dir)   # 期望返回: dict[name -> pose_dict]
    print("Reading ground truth poses...")
    gt   = load_poses(args.gt_dir)     # 期望返回: dict[name -> pose_dict]

    # =======【这里是关键修改】=======
    # 过去的代码是：
    # pred_map = {v.name: k for k,v in pred.items()}
    # gt_map   = {v.name: k for k,v in gt.items()}
    # 这要求 pred / gt 的值是“带 .name 属性的对象”，与你现在的返回结构不一致。
    # 我们改成：直接把“文件名作为 key”的 dict 做交集，不再做 map。
    # 如果你想按 basename 匹配，就把 key 先转成 basename。
    if args.match == "basename":
        pred = {os.path.basename(k): v for k, v in pred.items()}
        gt   = {os.path.basename(k):   v for k, v in gt.items()}
    # =======【修改结束】=======

    # 求公共文件名
    names_pred = set(pred.keys())
    names_gt   = set(gt.keys())
    common     = sorted(names_pred & names_gt)

    print(f"Common images: {len(common)} / pred={len(pred)} gt={len(gt)}")
    if len(common) == 0:
        print("No common images between pred and gt.")
        sys.exit(1)

    #testing
    import numpy as np
    for name in common:
        p = pred[name]
        g = gt[name]
        nq = np.linalg.norm(p.get('qvec_wxyz', [0,0,0,0]))
        ng = np.linalg.norm(g.get('qvec_wxyz', [0,0,0,0]))
        if nq == 0 or ng == 0:
            print(f"[debug] zero-quat at {name}: pred_nq={nq}, gt_nq={ng}")
            print("       pred_q:", p.get('qvec_wxyz'))
            print("       gt_q:",   g.get('qvec_wxyz'))

    #testing end

    # 将公共名对应的位姿转成 4x4 c2w
    pred_Ts, gt_Ts = [], []
    for name in common:
        pred_Ts.append(imgpose_to_c2w(pred[name]))  # 注意: pred[name] 是 dict，见下方提醒
        gt_Ts.append(imgpose_to_c2w(gt[name]))



    # ===== 全面诊断：同时在 c2w / w2c 求 S，并尝试多种应用方式，选旋转ATE最低的 =====
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    # 取 c2w / w2c 旋转阵列
    R_pred_c2w = np.stack([T[:3, :3] for T in pred_Ts], axis=0)
    R_gt_c2w   = np.stack([T[:3, :3] for T in gt_Ts],   axis=0)
    R_pred_w2c = np.transpose(R_pred_c2w, (0, 2, 1))
    R_gt_w2c   = np.transpose(R_gt_c2w,   (0, 2, 1))

    def wahba_S(A_list, B_list):
        # 求 S 使 A_i ≈ S B_i （左乘形式）
        A = np.zeros((3,3), dtype=np.float64)
        for Ai, Bi in zip(A_list, B_list):
            A += Ai @ Bi.T
        U, _, Vt = np.linalg.svd(A)
        S = U @ Vt
        if np.linalg.det(S) < 0:
            U[:, -1] *= -1
            S = U @ Vt
        return S

    def rot_ate_deg(Rs_pred, Rs_gt):
        errs = []
        for Rp, Rg in zip(Rs_pred, Rs_gt):
            Rrel = Rg.T @ Rp
            ang = np.degrees(np.arccos(np.clip((np.trace(Rrel)-1)/2, -1, 1)))
            errs.append(ang)
        return float(np.sqrt(np.mean(np.array(errs)**2)))

    # 1) 在 w2c 空间解：R_pred_w2c ≈ S_w2c * R_gt_w2c
    S_w2c = wahba_S(R_pred_w2c, R_gt_w2c)
    # 2) 在 c2w 空间解：R_pred_c2w ≈ R_gt_c2w * S_c2w   （等价于右乘）
    #    推导：R_pred_c2w ≈ R_gt_c2w * S  <=> R_pred_w2c ≈ S^T * R_gt_w2c
    B = np.zeros((3,3), dtype=np.float64)
    for Rp, Rg in zip(R_pred_c2w, R_gt_c2w):
        B += Rg.T @ Rp
    U, _, Vt = np.linalg.svd(B)
    S_c2w = U @ Vt
    if np.linalg.det(S_c2w) < 0:
        U[:, -1] *= -1
        S_c2w = U @ Vt

    # 尝试四种“把 S 应用回预测”的方式：
    # A) c2w 右乘 S_c2w：  R'^c2w = R^c2w * S_c2w
    R_pred_c2w_A = np.einsum('nij,jk->nik', R_pred_c2w, S_c2w)
    # B) c2w 左乘 S_w2c^{-T}：利用 w2c 的 S，转到 c2w 的右乘等价（S_w2c^{-T} = (S_w2c^T)^{-1} = S_w2c）
    #    推导：R_w2c' = S_w2c R_w2c  => R_c2w' = (R_w2c')^T = R_c2w S_w2c^T
    R_pred_c2w_B = np.einsum('nij,jk->nik', R_pred_c2w, S_w2c.T)
    # C) w2c 左乘 S_w2c：  R'^w2c = S_w2c * R^w2c
    R_pred_w2c_C = np.einsum('ij,njk->nik', S_w2c, R_pred_w2c)
    # D) w2c 右乘 S_c2w^{-1}：R'^w2c = R^w2c * S_c2w^{-1}
    R_pred_w2c_D = np.einsum('nij,jk->nik', R_pred_w2c, S_c2w.T)  # 因 S_c2w 是正交阵，逆=转置

    # 计算每种方式的旋转ATE（与对应GT在相同空间比）
    ate_A = rot_ate_deg(R_pred_c2w_A, R_gt_c2w)
    ate_B = rot_ate_deg(R_pred_c2w_B, R_gt_c2w)
    ate_C = rot_ate_deg(R_pred_w2c_C, R_gt_w2c)
    ate_D = rot_ate_deg(R_pred_w2c_D, R_gt_w2c)

    print("[diagnose] Rotation ATE candidates (deg): "
          f"A c2w*Sc2w={ate_A:.3f}, B c2w*Sw2c^T={ate_B:.3f}, "
          f"C Sw2c*w2c={ate_C:.3f}, D w2c*Sc2w^-1={ate_D:.3f}")

    # 选择最小的一个，并构造“修正后的 c2w 轨迹”用于后续对比（不影响你原有输出）
    which = np.argmin([ate_A, ate_B, ate_C, ate_D])
    if which == 0:
        best_mode = "A_c2w_rightmul_Sc2w"; R_pred_c2w_best = R_pred_c2w_A
    elif which == 1:
        best_mode = "B_c2w_rightmul_Sw2cT"; R_pred_c2w_best = R_pred_c2w_B
    elif which == 2:
        best_mode = "C_w2c_leftmul_Sw2c";   R_pred_c2w_best = np.transpose(R_pred_w2c_C, (0,2,1))
    else:
        best_mode = "D_w2c_rightmul_Sc2wT"; R_pred_c2w_best = np.transpose(R_pred_w2c_D, (0,2,1))

    print(f"[diagnose] best mode = {best_mode}")

    # 构造 pred_Ts_rotfixed（只修旋转，不动平移）
    pred_Ts_rotfixed = []
    for T, Rfix in zip(pred_Ts, R_pred_c2w_best):
        Afix = np.eye(4)
        Afix[:3,:3] = Rfix
        Afix[:3, 3] = T[:3, 3]
        pred_Ts_rotfixed.append(Afix)

    # 打印原始与修正的旋转 ATE（c2w 空间）
    ate_r_deg_orig     = ate_rot(pred_Ts, gt_Ts)
    ate_r_deg_rotfixed = ate_rot(pred_Ts_rotfixed, gt_Ts)
    print(f"[diagnose] Rotation ATE (orig)     : {ate_r_deg_orig:.3f} deg")
    print(f"[diagnose] Rotation ATE (rotfixed) : {ate_r_deg_rotfixed:.3f} deg")
    # ===== 诊断结束 =====

    # ——相邻帧“相对旋转误差”（度）——
    adj_errs = []
    for i in range(len(pred_Ts) - 1):
        Rp_i = pred_Ts[i][:3, :3];
        Rp_j = pred_Ts[i + 1][:3, :3]
        Rg_i = gt_Ts[i][:3, :3];
        Rg_j = gt_Ts[i + 1][:3, :3]
        Rrel_p = Rp_i.T @ Rp_j
        Rrel_g = Rg_i.T @ Rg_j
        Rerr = Rrel_g.T @ Rrel_p
        ang = np.degrees(np.arccos(np.clip((np.trace(Rerr) - 1) / 2, -1, 1)))
        adj_errs.append(float(ang))
    if adj_errs:
        import numpy as np
        arr = np.array(adj_errs)
        print(
            f"[diagnose] adjacent relative-rot err: mean={arr.mean():.2f}°, median={np.median(arr):.2f}°, 90p={np.percentile(arr, 90):.2f}° (N={len(arr)})")
    # ——相邻帧“相对旋转误差”（度）——


    # SE3 对齐
    Rg, tg = se3_align(pred_Ts, gt_Ts)
    pred_se3 = []
    for T in pred_Ts:
        A = np.eye(4)
        A[:3, :3] = Rg @ T[:3, :3]
        A[:3, 3]  = Rg @ T[:3, 3] + tg
        pred_se3.append(A)

    # Sim3 对齐
    Rgs, tgs, s = sim3_align(pred_Ts, gt_Ts)
    pred_sim3 = []
    for T in pred_Ts:
        A = np.eye(4)
        A[:3, :3] = Rgs @ T[:3, :3]
        A[:3, 3]  = Rgs @ (T[:3, 3] * s) + tgs
        pred_sim3.append(A)

    # ===== 指标 =====
    ate_t_no_scale = ate_trans(pred_se3, gt_Ts)
    ate_t_scale    = ate_trans(pred_sim3, gt_Ts)
    ate_r_deg      = ate_rot(pred_se3, gt_Ts)
    rte_t_m        = rte_trans_allpairs(pred_se3, gt_Ts)
    rte_t_deg, tdeg_arr = rte_trans_angle_allpairs(pred_se3, gt_Ts)
    rte_r_deg, rdeg_arr = rte_rot_allpairs(pred_se3, gt_Ts)
    AUC30          = auc30(rdeg_arr, tdeg_arr, 30)

    # 完整度（建议用“公共张数/GT总数”，更符合覆盖率直觉）
    completeness   = len(common) / max(1, len(gt))

    with open(args.output, "w") as f:
        f.write("Pose Estimation Evaluation Results\n")
        f.write("="*50 + "\n\n")
        f.write("Reconstruction Completeness:\n")
        f.write(f"  Number of predicted images: {len(pred)}\n")
        f.write(f"  Number of ground truth images: {len(gt)}\n")
        f.write(f"  Number of common images: {len(common)}\n")
        f.write(f"  Completeness ratio: {completeness:.4f}\n\n")
        f.write("Absolute Trajectory Error (ATE):\n")
        f.write(f"  Translation ATE (no scale): {ate_t_no_scale:.6f} m\n")
        f.write(f"  Translation ATE (scale):    {ate_t_scale:.6f} m\n")
        f.write(f"  Rotation ATE:               {ate_r_deg:.6f} deg\n\n")
        f.write("Relative Trajectory Error (RTE):\n")
        f.write(f"  Translation RTE:            {rte_t_m:.6f} m\n")
        f.write(f"  Translation RTE (deg):      {rte_t_deg:.6f} deg\n")
        f.write(f"  Rotation RTE (deg):         {rte_r_deg:.6f} deg\n\n")
        f.write("AUC@30:\n")
        f.write(f"  AUC@30:                     {AUC30:.6f}\n\n")
        f.write("Alignment Params:\n")
        f.write(f"  Scale factor:               {s:.6f}\n")
        f.write(f"  SE3 t:                      {tg.tolist()}\n")

    print("Done. Results ->", args.output)


if __name__ == "__main__":
    main()
