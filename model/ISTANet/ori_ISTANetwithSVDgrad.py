import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def peak_loss(S_pred, S_true, cor_sources, T=0.1):
    power_pred = S_pred.pow(2).mean(dim=-1) / T
    power_true = S_true.pow(2).mean(dim=-1) / T

    w_pred = F.softmax(power_pred, dim=-1)
    w_true = F.softmax(power_true, dim=-1)

    peak_pred = (w_pred.unsqueeze(-1) * cor_sources).sum(dim=1)
    peak_true = (w_true.unsqueeze(-1) * cor_sources).sum(dim=1)

    return torch.norm(peak_pred - peak_true, dim=-1).mean()


class DirectWGradStep(nn.Module):
    def __init__(self, L: torch.Tensor, k: int = 128, alpha: float = 0.1, eps: float = 1e-8):
        super().__init__()
        U, S, Vt = torch.linalg.svd(L, full_matrices=False)
        k = min(k, S.numel())

        self.eps = eps
        self.register_buffer('P', U[:, :k].contiguous(), persistent=False)
        self.register_buffer('Lambda', S[:k].contiguous(), persistent=False)
        self.register_buffer('Q', Vt[:k, :].contiguous(), persistent=False)

        self.lambda_i = nn.Parameter(torch.full((k,), float(alpha), dtype=S.dtype, device=S.device))
        self.w = nn.Parameter(torch.ones(k, dtype=S.dtype, device=S.device))

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        Y_tilde = torch.matmul(self.P.T, Y)
        lambda_i = self.lambda_i.clamp_min(self.eps)
        f_i = self.w * self.Lambda / (self.Lambda ** 2 + lambda_i)
        S_tilde = f_i.view(1, -1, 1) * Y_tilde
        return torch.matmul(self.Q.T, S_tilde)




class Phi(nn.Module):
    def __init__(self, n_source: int, Phi_1_dim: int, Phi_2_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(n_source, Phi_1_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv1d(Phi_1_dim, Phi_2_dim, kernel_size=1, padding=0, bias=False),
        )
        for m in self.fc.modules():
            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)



class PhiTilde(nn.Module):
    def __init__(self, n_source: int, Phi_1_dim: int, Phi_2_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(Phi_2_dim, Phi_1_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv1d(Phi_1_dim, n_source, kernel_size=1, padding=0, bias=False),
        )
        for m in self.fc.modules():
            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# =========================================================================
# 4. 核心模块: ISTA Block
# =========================================================================

class ISTABlock(nn.Module):
    def __init__(self, n_source, Phi_1_dim: int = 500, Phi_2_dim: int = 128):
        super().__init__()

        self.soft_thr_spatial = nn.Parameter(torch.full((1, Phi_2_dim, 1), -4.6))

        self.Phi = Phi(n_source, Phi_1_dim, Phi_2_dim)
        self.Phi_tilde = PhiTilde(n_source, Phi_1_dim, Phi_2_dim)

        self.relu = nn.ReLU()

    def forward(self, x_grad):
        feat = self.Phi(x_grad)
        x_shrink = torch.sign(feat) * self.relu(torch.abs(feat) - F.softplus(self.soft_thr_spatial))
        x_pred = self.Phi_tilde(x_shrink)
        return x_pred


class ISTA(nn.Module):
    def __init__(self,
                 L,
                 LayerNo,
                 n_source,
                 M_matrix,
                 cor_sources,
                 truned_channels,
                 alpha_loss=0.1,
                 beta_loss=0.1,
                 grad_step_alpha=0.1,
                 Phi_1_dim=500,
                 Phi_2_dim=128,
                 eps=1e-8):
        super().__init__()
        self.LayerNo = LayerNo
        self.eps = eps

        self.register_buffer('L', L, persistent=False)
        self.register_buffer('M_matrix', M_matrix, persistent=False)
        self.register_buffer('cor_sources', cor_sources, persistent=False)

        self.grad_step = DirectWGradStep(L, k=truned_channels, alpha=grad_step_alpha, eps=eps)
        self._truned_channels = self.grad_step.w.numel()

        self.blocks = nn.ModuleList([
            ISTABlock(n_source, Phi_1_dim=Phi_1_dim, Phi_2_dim=Phi_2_dim)
            for _ in range(LayerNo)
        ])

        self.alpha = alpha_loss
        self.beta = beta_loss

    def forward(self, eeg, return_all=False):
        S = self.grad_step(eeg)

        S_list = []
        for blk in self.blocks:
            S = blk(S)
            S_list.append(S)

        if return_all:
            return S_list
        return S

    def compute_loss(self, Y_t, S_true):
        S_list = self.forward(Y_t, return_all=True)
        S_pred = S_list[-1]


        loss_d = torch.mean(torch.sum((S_pred - S_true) ** 2, dim = -2))
        # loss_d = torch.mean((S_pred - S_true) ** 2)

        loss_peak = peak_loss(S_pred, S_true, self.cor_sources)

        M_S_true = torch.einsum('ij,bjt->bit', self.M_matrix, S_true)
        M_S_pred = torch.einsum('ij,bjt->bit', self.M_matrix, S_pred)
        loss_t = torch.mean(torch.sum((M_S_true - M_S_pred) ** 2, dim=-2))

        loss_all = loss_d + self.alpha * loss_peak + self.beta * loss_t
        # loss_all = self.alpha * loss_peak + self.beta * loss_t

        loss_dict = {
            'loss_d': loss_d.item(),
            'loss_peak': loss_peak.item(),
            'loss_t': loss_t.item(),
            'loss_all': loss_all.item(),
            'dbg_S_true_abs_max': S_true.detach().abs().max().item(),
            'dbg_S_pred_abs_max': S_pred.detach().abs().max().item(),
            'dbg_Yt_abs_max': Y_t.detach().abs().max().item(),
            'dbg_w_min': self.grad_step.w.detach().min().item(),
            'dbg_w_max': self.grad_step.w.detach().max().item(),
            'dbg_w_mean': self.grad_step.w.detach().mean().item(),
            'dbg_lambda_min': self.grad_step.lambda_i.detach().min().item(),
            'dbg_lambda_max': self.grad_step.lambda_i.detach().max().item(),
            'dbg_lambda_mean': self.grad_step.lambda_i.detach().mean().item(),
            'dbg_thr_min': min(F.softplus(blk.soft_thr_spatial.detach()).min().item() for blk in self.blocks),
            'dbg_thr_max': max(F.softplus(blk.soft_thr_spatial.detach()).max().item() for blk in self.blocks),
        }

        return loss_all, loss_dict

    def predict(self, Y_ob):
        return self.forward(Y_ob)
