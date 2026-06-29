import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import roc_curve, auc


class EvaluatorTorch:
    """
    基于 PyTorch 的加速评估器：
    - 优势：当使用 GPU（CUDA）时，SD/DLE 的大规模距离计算（[B,K,N]）可显著加速；
    - 做法：在 no_grad + detach 下，将数据转为 torch.Tensor，尽量在同一 device 上计算，最后再转回 CPU 标量。

    注意：若仅在 CPU 上跑，和 NumPy 相比性能差异不一定明显；是否更快取决于 B/N/K/T 规模与内存带宽。
    """

    def __init__(self, lead_tensor, cor_sources_tensor, count_source_power_greater_than:float = 0.1):
        self.lead_tensor = lead_tensor
        self.count_source_power_greater_than = count_source_power_greater_than

        self.cor_sources = cor_sources_tensor  # used by SD/DLE routines
        self.device = cor_sources_tensor.device
        self.dtype = cor_sources_tensor.dtype
        # 源坐标（mm）放到 device
        # cor_np = (self.fwd['source_rr'] * 1e3).astype(np.float32)
        # self.cor_sources = torch.from_numpy(cor_np).to(self.device, dtype=self.dtype)  # [N,3]

    # -----------------------
    # 统一输入 & 广播工具
    # -----------------------
    @staticmethod
    def _to_tensor(x, device, dtype):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().to(device=device, dtype=dtype, non_blocking=True)
        x = np.asarray(x)
        return torch.from_numpy(x).to(device=device, dtype=dtype)

    @staticmethod
    def _normalize_true_cor_shape(true_cor: torch.Tensor, B: int) -> torch.Tensor:
        # 期望输出 [B, K, 3]
        if true_cor.dim() == 1 and true_cor.shape[0] == 3:
            true_cor = true_cor.view(1, 1, 3)
        elif true_cor.dim() == 2 and true_cor.shape[1] == 3:
            if true_cor.shape[0] == B:
                true_cor = true_cor.view(B, 1, 3)
            else:
                true_cor = true_cor.unsqueeze(0).expand(B, -1, 3)
        elif true_cor.dim() == 3 and true_cor.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported true_cor shape: {tuple(true_cor.shape)}")
        if true_cor.shape[0] == 1 and B > 1:
            true_cor = true_cor.expand(B, -1, -1)
        return true_cor

    # -----------------------
    # 公共接口
    # -----------------------
    def evaluate(self, pred, true_cor=None, true_sig=None, auc_mode: str = 'macro') -> dict:
        with torch.no_grad():
            pred_t = self._to_tensor(pred, self.device, self.dtype)
            if pred_t.dim() == 2:
                pred_t = pred_t.unsqueeze(0)
            if pred_t.dim() != 3:
                raise ValueError(f"pred shape expected [B,N,T] or [N,T], got {tuple(pred_t.shape)}")
            B = pred_t.shape[0]

            true_sig_t = self._to_tensor(true_sig, self.device, self.dtype)
            if true_sig_t is not None:
                true_sig_t = self.filter_source(true_sig_t, threshold_rate=self.count_source_power_greater_than)
                if true_sig_t.dim() == 2:
                    true_sig_t = true_sig_t.unsqueeze(0)
                if true_sig_t.dim() != 3:
                    raise ValueError(f"true_sig shape expected [B,N,T] or [N,T], got {tuple(true_sig_t.shape)}")

            true_cor_t = self._to_tensor(true_cor, self.device, self.dtype)
            if true_cor_t is not None:
                true_cor_t = self._normalize_true_cor_shape(true_cor_t, B)

            SD = self.compute_SpatialDispersion(pred_t, true_S=true_sig_t, true_Cor=true_cor_t, filter_threshold_rate=self.count_source_power_greater_than).mean()
            DLE = self.compute_DistanceLocalizationError(pred_t, true_S=true_sig_t, true_Cor=true_cor_t, filter_threshold_rate=self.count_source_power_greater_than).mean()
            try:
                AUC = self.compute_AUC(pred_t, true_S=true_sig_t, true_Cor=true_cor_t, mode=auc_mode, filter_threshold_rate=self.count_source_power_greater_than)
            except Exception:
                AUC = float('nan')
            out = {
                'SD': float(SD.detach().cpu().item()),
                'DLE': float(DLE.detach().cpu().item()),
                'AUC': float(AUC) if (isinstance(AUC, float) or np.isscalar(AUC)) else float(AUC)
            }
            if true_sig_t is not None:
                SE = self.compute_ShapeError(pred_t, true_sig_t).mean()
                SE = float(SE.detach().cpu().item())
                out['SE'] = SE if np.isfinite(SE) else float('nan')
            return out

    # -----------------------
    # 过滤弱源（向量化）
    # -----------------------
    @staticmethod
    def filter_source(y_pred: torch.Tensor, threshold_rate: float = 0.1) -> torch.Tensor:
        # y_pred: [B,N,T]
        source_powers = y_pred.abs().sum(dim=-1)                 # [B,N]
        max_power = source_powers.max(dim=1, keepdim=True).values
        thr = threshold_rate * max_power
        mask = (source_powers >= thr)                             # [B,N]
        return y_pred * mask.unsqueeze(-1)

    # -----------------------
    # SD
    # -----------------------
    def compute_SpatialDispersion(self, pred_S: torch.Tensor, true_S: torch.Tensor | None = None, true_Cor: torch.Tensor | None = None, filter_threshold_rate: float = 0.1) -> torch.Tensor:
        with torch.no_grad():
            pred_S = self.filter_source(pred_S, threshold_rate=filter_threshold_rate)
            q_hat = (pred_S * pred_S).sum(dim=-1)   # [B,N]
            denom = q_hat.sum(dim=1)                # [B]

            B, N, _ = pred_S.shape

            if true_Cor is not None:
                # D: [B,K,N]
                # 批量 cdist：x [B,K,3], y [B,N,3]
                y = self.cor_sources.unsqueeze(0).expand(B, -1, -1)  # [B,N,3]
                D = torch.cdist(true_Cor, y, p=2)                     # [B,K,N]
                nearest_k = D.argmin(dim=1)                           # [B,N]
                D_sq = D.square()
                D_sel = torch.gather(D_sq, dim=1, index=nearest_k.unsqueeze(1)).squeeze(1)  # [B,N]
                num = (D_sel * q_hat).sum(dim=1)
                SD = torch.zeros_like(denom)
                safe = denom > 1e-12
                SD[safe] = (num[safe] / denom[safe]).sqrt()
                return SD

            if true_S is not None:
                # K 可变，逐样本处理，但 cdist 仍在 GPU 上
                SD = torch.zeros(B, device=self.device, dtype=self.dtype)
                for b in range(B):
                    q_true = (true_S[b] * true_S[b]).sum(dim=-1)           # [N]
                    act_idx = torch.nonzero(q_true > 1e-10, as_tuple=False).flatten()
                    if act_idx.numel() == 0:
                        SD[b] = 0.0
                        continue
                    tc = self.cor_sources.index_select(0, act_idx).unsqueeze(0)   # [1,K,3]
                    y = self.cor_sources.unsqueeze(0)                             # [1,N,3]
                    D = torch.cdist(tc, y, p=2).squeeze(0)                        # [K,N]
                    nearest_k = D.argmin(dim=0)                                    # [N]
                    D_sel = D.square().gather(dim=0, index=nearest_k.unsqueeze(0)).squeeze(0)
                    denom_b = denom[b]
                    if denom_b <= 1e-12:
                        SD[b] = 0.0
                    else:
                        num_b = (D_sel * q_hat[b]).sum()
                        SD[b] = (num_b / denom_b).sqrt()
                return SD

            raise ValueError("Either true_S or true_Cor must be provided.")

    # -----------------------
    # DLE
    # -----------------------
    def compute_DistanceLocalizationError(self, pred_S: torch.Tensor, true_S: torch.Tensor | None = None, true_Cor: torch.Tensor | None = None, filter_threshold_rate: float = 0.1) -> torch.Tensor:
        with torch.no_grad():
            pred_S = self.filter_source(pred_S, threshold_rate=filter_threshold_rate)
            q_hat = (pred_S * pred_S).sum(dim=-1)  # [B,N]
            B, N, _ = pred_S.shape

            if true_Cor is not None:
                y = self.cor_sources.unsqueeze(0).expand(B, -1, -1)  # [B,N,3]
                D = torch.cdist(true_Cor, y, p=2)                     # [B,K,N]
                K = D.shape[1]
                nearest_k = D.argmin(dim=1)                           # [B,N]
                # energies_clusters: [B,K,N]
                Ks = torch.arange(K, device=self.device).view(1, K, 1)
                energies_clusters = torch.where(
                    nearest_k.unsqueeze(1) == Ks,
                    q_hat.unsqueeze(1),
                    torch.full_like(q_hat.unsqueeze(1), float('-inf'))
                )
                max_energy, i_star = energies_clusters.max(dim=2)      # [B,K]
                has_sources = torch.isfinite(max_energy)               # [B,K]
                # DLE_k = D[b,k,i*]
                DLE_k = D.gather(dim=2, index=i_star.unsqueeze(-1)).squeeze(-1)  # [B,K]
                # 平均（仅对有源的 k）
                counts = has_sources.sum(dim=1)                        # [B]
                counts = torch.clamp_min(counts, 1)
                dle_sum = (DLE_k * has_sources).sum(dim=1)
                return (dle_sum / counts)

            if true_S is not None:
                DLE = torch.zeros(B, device=self.device, dtype=self.dtype)
                for b in range(B):
                    q_true = (true_S[b] * true_S[b]).sum(dim=-1)
                    act_idx = torch.nonzero(q_true > 1e-10, as_tuple=False).flatten()
                    if act_idx.numel() == 0:
                        DLE[b] = 0.0
                        continue
                    tc = self.cor_sources.index_select(0, act_idx).unsqueeze(0)  # [1,K,3]
                    y = self.cor_sources.unsqueeze(0)                            # [1,N,3]
                    D_b = torch.cdist(tc, y, p=2).squeeze(0)                     # [K,N]
                    nearest_k = D_b.argmin(dim=0)                                 # [N]
                    vals = []
                    for k in range(D_b.shape[0]):
                        idx_i = torch.nonzero(nearest_k == k, as_tuple=False).flatten()
                        if idx_i.numel() == 0:
                            continue
                        local_q = q_hat[b].index_select(0, idx_i)
                        i_star_local = idx_i[torch.argmax(local_q)]
                        vals.append(D_b[k, i_star_local])
                    DLE[b] = torch.stack(vals).mean() if len(vals) > 0 else torch.tensor(0.0, device=self.device, dtype=self.dtype)
                return DLE

            raise ValueError("Either true_S or true_Cor must be provided.")

    # -----------------------
    # AUC（仍用 sklearn 计算曲线与面积）
    # -----------------------
    def compute_AUC(self, pred_S, true_S=None, true_Cor=None, filter_threshold_rate: float = 0.1, mode: str = 'macro'):
        with torch.no_grad():
            pred_S = self._to_tensor(pred_S, self.device, self.dtype)
            if pred_S.dim() != 3:
                raise ValueError("pred_S must have shape [B,N,T].")
            B, N, _ = pred_S.shape

            pred_S = self.filter_source(pred_S, threshold_rate=filter_threshold_rate)
            pred_energies = torch.linalg.vector_norm(pred_S, dim=2).cpu().numpy()  # [B,N]

            if true_S is not None:
                true_S = self._to_tensor(true_S, self.device, self.dtype)
                true_energies = torch.linalg.vector_norm(true_S, dim=2).cpu().numpy()
                labels = (true_energies > 1e-10).astype(np.uint8)
            elif true_Cor is not None:
                true_Cor = self._to_tensor(true_Cor, self.device, self.dtype)
                true_Cor = self._normalize_true_cor_shape(true_Cor, B)
                corr = self.cor_sources.unsqueeze(0)  # [1,N,3]
                # 计算 [B,K,N] 距离
                y = corr.expand(B, -1, -1)
                D = torch.cdist(true_Cor, y, p=2)
                nn_idx = torch.argmin(D, dim=1)  # [B,N] 每列为对应 k 的最近 N? 注意，这里我们想要每个 k 的 argmin over N
                # 上一行返回 [B,K] 最近顶点索引，实际上 argmin(dim=2)
                nn_idx = torch.argmin(D, dim=2)  # [B,K]
                labels = np.zeros((B, N), dtype=np.uint8)
                for b in range(B):
                    labels[b, nn_idx[b].cpu().numpy()] = 1
            else:
                raise ValueError("Either true_S or true_Cor must be provided.")

            if mode == 'micro':
                y_true = labels.ravel()
                y_score = pred_energies.ravel()
                finite = np.isfinite(y_score)
                if not finite.all():
                    y_true = y_true[finite]
                    y_score = y_score[finite]
                if y_true.size == 0:
                    return float('nan')
                if y_true.sum() == 0 or y_true.sum() == y_true.size:
                    return float('nan')
                fpr, tpr, _ = roc_curve(y_true, y_score)
                return auc(fpr, tpr)
            elif mode == 'macro':
                auc_list = []
                for b in range(B):
                    y_true = labels[b]
                    y_score = pred_energies[b]
                    finite = np.isfinite(y_score)
                    if not finite.all():
                        y_true = y_true[finite]
                        y_score = y_score[finite]
                    if y_true.size == 0:
                        continue
                    if y_true.sum() == 0 or y_true.sum() == y_true.size:
                        continue
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    auc_list.append(auc(fpr, tpr))
                return float(np.nanmean(auc_list)) if auc_list else float('nan')
            else:
                raise ValueError("mode must be 'micro' or 'macro'")

    # -----------------------
    # Shape Error
    # -----------------------
    @staticmethod
    def compute_ShapeError(pred_S: torch.Tensor, true_S: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        with torch.no_grad():
            if pred_S.dim() != 3 or true_S.dim() != 3:
                raise ValueError("pred_S and true_S must have shape [B,N,T].")
            if pred_S.shape != true_S.shape:
                raise ValueError(f"Shape mismatch: pred_S {tuple(pred_S.shape)} vs true_S {tuple(true_S.shape)}")
            B = pred_S.shape[0]
            pred_flat = pred_S.reshape(B, -1)
            true_flat = true_S.reshape(B, -1)
            nPh = torch.linalg.vector_norm(pred_flat, dim=1)
            nTr = torch.linalg.vector_norm(true_flat, dim=1)
            nPh_safe = torch.clamp(nPh, min=eps)
            nTr_safe = torch.clamp(nTr, min=eps)
            zero_both = (nPh < eps) & (nTr < eps)
            diff = pred_flat / nPh_safe.unsqueeze(1) - true_flat / nTr_safe.unsqueeze(1)
            se = (diff * diff).sum(dim=1)
            se[zero_both] = 0.0
            return se
