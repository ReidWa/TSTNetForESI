import os
from datetime import datetime

import numpy as np
import torch
from scipy.sparse import coo_matrix, block_diag
from scipy.sparse.csgraph import dijkstra

from utils.utils_train import load_forward_fixed


date_str = datetime.now().strftime("%Y%m%d_%H%M%S")


BUILD_M_CONFIG = {
    # Windows 路径
    # "fwd_path": r"E:\2_ESIdiff\Dataset\localize-mi-data\derivatives\sourcemodelling\sub-01\fwd\sub-01_fwd.fif",
    # "save_path": rf"E:\2_ESIdiff\results\ESIdiff\M_matrix_{date_str}.pt",

    # Linux 路径
    "fwd_path": "/mnt/hdd/data/EEG/MI-ME/Wangkx_privated/esi/fwd/sub-01_fwd.fif",
    "save_path": f"/mnt/disk16t/checkpoints/ESIdiff/M_matrix_{date_str}.pt",

    # AUTO DL 路径
    # "fwd_path": "/root/autodl-tmp/sub-01/fwd/sub-01_fwd.fif",
    # "save_path": f"/root/autodl-fs/checkpoints/M_matrix_{date_str}.pt",

    "sigma_mm": 20,
    "limit_mm": 35,
}


def compute_hemi_M(hemi_src, sigma=0.015, dist_limit=0.020):
    """计算单侧半球的拓扑矩阵 M。"""
    rr = hemi_src['rr']
    tris = hemi_src['tris']
    vertno = hemi_src['vertno']

    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    weights = np.linalg.norm(rr[edges[:, 0]] - rr[edges[:, 1]], axis=1)

    n_vertices = rr.shape[0]
    adj = coo_matrix(
        (
            np.hstack([weights, weights]),
            (
                np.hstack([edges[:, 0], edges[:, 1]]),
                np.hstack([edges[:, 1], edges[:, 0]]),
            ),
        ),
        shape=(n_vertices, n_vertices),
    ).tocsr()

    dist_all = dijkstra(adj, directed=False, indices=vertno)
    dist_matrix = dist_all[:, vertno]

    M = np.zeros_like(dist_matrix, dtype=np.float32)
    mask = dist_matrix <= dist_limit
    M[mask] = np.exp(-((dist_matrix[mask] / sigma) ** 2)).astype(np.float32)
    return M


def build_M_matrix(fwd, sigma_mm=15, limit_mm=20):
    """从 fwd 文件构建完整的 M 矩阵。"""
    sigma = sigma_mm / 1000.0
    dist_limit = limit_mm / 1000.0

    src = fwd['src']
    if src[0]['type'] != 'surf':
        raise ValueError("只有表面源空间(surf)才能计算皮层测地线拓扑！")

    print("正在计算左半球拓扑...")
    M_lh = compute_hemi_M(src[0], sigma, dist_limit)

    print("正在计算右半球拓扑...")
    M_rh = compute_hemi_M(src[1], sigma, dist_limit)

    M_full = block_diag((M_lh, M_rh)).toarray().astype(np.float32)
    return torch.from_numpy(M_full)


def main():
    config = BUILD_M_CONFIG

    print("=" * 50)
    print("加载 Forward Model...")
    fwd = load_forward_fixed(config["fwd_path"])

    print("=" * 50)
    print("构建 M_matrix...")
    M_matrix = build_M_matrix(
        fwd,
        sigma_mm=config["sigma_mm"],
        limit_mm=config["limit_mm"],
    )

    save_path = config["save_path"]
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            "M_matrix": M_matrix,
            "sigma_mm": config["sigma_mm"],
            "limit_mm": config["limit_mm"],
            "shape": tuple(M_matrix.shape),
            "fwd_path": config["fwd_path"],
        },
        save_path,
    )

    print("=" * 50)
    print(f"M_matrix shape: {tuple(M_matrix.shape)}")
    print(f"已保存到: {save_path}")


if __name__ == "__main__":
    main()
