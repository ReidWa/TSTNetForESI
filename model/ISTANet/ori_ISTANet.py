import torch
import torch.nn as nn
from torch.nn import init


class InitS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, EEG: torch.Tensor, LT: torch.Tensor) -> torch.Tensor:
        return torch.matmul(LT, EEG)


class Phi(nn.Module):
    def __init__(self, n_source: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(n_source, 500, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv1d(500, 128, kernel_size=1, padding=0, bias=False),
        )
        for m in self.fc.modules():
            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class PhiTilde(nn.Module):
    def __init__(self, n_source: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(128, 500, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv1d(500, n_source, kernel_size=1, padding=0, bias=False),
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
    def __init__(self, n_source):
        super().__init__()

        self.rho = nn.Parameter(torch.tensor(0.1))
        self.soft_thr_spatial = nn.Parameter(torch.full((1, 128, 1), 0.01))

        self.Phi = Phi(n_source)
        self.Phi_tilde = PhiTilde(n_source)

        self.relu = nn.ReLU()

    def forward(self, x, L, LT, Y_obs):
        # R^(k) = S^(k-1) - rho * L^T (L S^(k-1) - Y)
        LS = torch.matmul(L, x)
        residual = LS - Y_obs
        grad = torch.matmul(LT, residual)
        x_grad = x - self.rho * grad

        feat = self.Phi(x_grad)
        x_shrink = torch.sign(feat) * self.relu(torch.abs(feat) - self.soft_thr_spatial)
        x_pred = self.Phi_tilde(x_shrink)
        return x_pred


class ISTA(nn.Module):
    def __init__(self,
                 L,
                 LayerNo,
                 n_source,
                 M_matrix,
                 alpha_loss=0.1,
                 beta_loss=0.1,
                 eps=1e-8):
        super().__init__()
        self.LayerNo = LayerNo
        self.eps = eps
        self.Q_init = InitS()

        self.blocks = nn.ModuleList([
            ISTABlock(n_source)
            for _ in range(LayerNo)
        ])

        self.register_buffer('L', L, persistent=False)
        self.register_buffer('LT', L.T.contiguous(), persistent=False)
        self.register_buffer('M_matrix', M_matrix, persistent=False)

        self.alpha = alpha_loss
        self.beta = beta_loss

    def forward(self, eeg, return_all=False):
        S = self.Q_init(eeg, self.LT)

        S_list = [S]
        for blk in self.blocks:
            S = blk(S, self.L, self.LT, eeg)
            S_list.append(S)

        if return_all:
            return S_list
        return S

    def compute_loss(self, Y_t, S_true):
        S_list = self.forward(Y_t, return_all=True)
        S_pred = S_list[-1]

        batch_size = Y_t.size(0)

        loss_d = torch.mean(torch.sum((S_pred - S_true) ** 2, dim = -2))

        loss_s = 0.0
        for k in range(1, len(S_list)):
            loss_s += torch.sum((S_list[k] - S_list[k - 1]) ** 2, dim = -2)
        loss_s = torch.mean(loss_s)

        M_S_true = torch.einsum('ij,bjt->bit', self.M_matrix, S_true)
        M_S_pred = torch.einsum('ij,bjt->bit', self.M_matrix, S_pred)
        loss_t = torch.mean(torch.sum((M_S_true - M_S_pred) ** 2, dim=-2))

        loss_all = loss_d + self.alpha * loss_s + self.beta * loss_t

        loss_dict = {
            'loss_d': loss_d.item(),
            'loss_s': loss_s.item(),
            'loss_t': loss_t.item(),
            'loss_all': loss_all.item(),
            'dbg_S_true_abs_max': S_true.detach().abs().max().item(),
            'dbg_S_pred_abs_max': S_pred.detach().abs().max().item(),
            'dbg_Yt_abs_max': Y_t.detach().abs().max().item(),
        }

        return loss_all, loss_dict

    def predict(self, Y_ob):
        return self.forward(Y_ob)
