import os
import random
from datetime import datetime

import numpy as np
import swanlab
import torch
from torch.utils.data import DataLoader

from data.dataset import get_trainsim_testsim_dataset
from Experiment.trainer import Trainer
from model.ISTANet.ori_ISTANetwithSVDgrad import ISTA as ori_ISTANet
from utils.utils_train import load_forward_fixed


date_str = datetime.now().strftime("%Y%m%d_%H%M%S")


TRAIN_CONFIG = {
    "project": "ori_ISTANet",
    "exp_name": f"ori_ISTANet_withSVDgradV0_4_PeaklossandPosThr_64trunproxy512-1000{date_str}",
    # Windows 路径
    # "data_path": r"E:\2_ESIdiff\Dataset\localize-mi-data\derivatives\simulationV2\trainset\collection_all\sub-01\runs_dataV2",
    # "fwd_path": r"E:\2_ESIdiff\Dataset\localize-mi-data\derivatives\sourcemodelling\sub-01\fwd\sub-01_fwd.fif",
    # "m_matrix_path": r"E:\2_ESIdiff\results\ESIdiff\M_matrix.pt",
    # "results_folder": rf"E:\2_ESIdiff\results\ESIdiff\ori_ISTANet_{date_str}",
    # Linux 路径
    "data_path": "/mnt/hdd/data/EEG/MI-ME/Wangkx_privated/esi/trainset/runs_data",
    "fwd_path": "/mnt/hdd/data/EEG/MI-ME/Wangkx_privated/esi/fwd/sub-01_fwd.fif",
    "m_matrix_path": "/mnt/hdd/data/EEG/MI-ME/Wangkx_privated/esi/fwd/M_matrix.pt",
    "results_folder": f"/mnt/disk16t/checkpoints/ESIdiff/ori_ISTANet_withSVDgradV0_4{date_str}",
    # AUTO DL 路径
    # "data_path": "/root/autodl-tmp/forautodl",
    # "fwd_path": "/root/autodl-tmp/sub-01/fwd/sub-01_fwd.fif",
    # "m_matrix_path": "/root/autodl-fs/checkpoints/M_matrix.pt",
    # "results_folder": f"/root/autodl-fs/checkpoints/ori_ISTANet/{date_str}",

    # 数据配置
    "data_preprocess": None,
    "num_runs": 20000,
    "train_rate": 0.95,
    "batch_size": 128,
    "eval_batch_size": 128,
    "num_workers": 8,
    "seed": 42,

    # 训练设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float32,

    # 训练参数
    "train_num_epochs": 200,
    "lr": 1e-4,
    "use_lr_scheduler": False,
    "lr_warmup_epochs": 5,
    "weight_decay": 1e-4,
    "grad_clip_norm": None,
    "save_and_sample_every_epoch": 1,
    "early_stop_patience": 200,

    # ori_ISTANet 配置
    "ISTANet_layers":1,
    "alpha_loss": 0.1,
    "beta_loss": 0.1,
    "truned_num": 64,
    "grad_step_alpha": 0.1,
    "Phi_1_dim": 512,
    "Phi_2_dim": 256,

    # 评估配置
    "eval_count_source_power_greater_than": 0.01,
    "eval_metric": "DLE",
    "eval_metric_mode": "min",

    # 继续训练配置
    "resume": False,
    "resume_checkpoint_path": None,

    # 自动推断
    "timepoints": None,
    "N_sources": None,
    "n_channel": None,

    # 日志
    "use_swanlab": True,
}


class _NullLogger:
    """空日志记录器，当不使用 swanlab 时使用"""

    def log(self, *args, **kwargs):
        return None


def seed_worker(worker_id: int):
    """确保每个 DataLoader worker 有确定性的 RNG 状态"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_global_seed(seed: int):
    """设置全局随机种子以确保可重复性"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_L(L, verbose=True):
    """
    归一化 leadfield 矩阵，使得 ||L||_F = sqrt(n_channels)
    这样当 S ~ N(0,1) 时，输出 Y = L @ S 的方差约为 1
    """
    n_channels, _ = L.shape
    current_f_norm = np.linalg.norm(L, ord='fro')
    target_f_norm = np.sqrt(n_channels)
    scaling_factor = target_f_norm / current_f_norm
    L_norm = L * scaling_factor

    if verbose:
        print(f"[L Normalization Info]")
        print(f"  Input Shape: {L.shape[0]} channels x {L.shape[1]} sources")
        print(f"  Target ||L||_F: {target_f_norm:.4f}")
        print(f"  Actual ||L||_F: {np.linalg.norm(L_norm, ord='fro'):.4f}")
        theoretical_out_var = (np.linalg.norm(L_norm, ord='fro') ** 2) / n_channels
        print(f"  Theoretical Output Variance: {theoretical_out_var:.4f} (Target: 1.0000)")

    return L_norm


def load_M_matrix(m_matrix_path: str, device: str) -> torch.Tensor:
    if not os.path.isfile(m_matrix_path):
        raise FileNotFoundError(f"未找到 M_matrix 文件: {m_matrix_path}")

    payload = torch.load(m_matrix_path, map_location="cpu")
    if isinstance(payload, dict):
        if "M_matrix" not in payload:
            raise KeyError(f"M_matrix 文件缺少键 'M_matrix': {m_matrix_path}")
        M_matrix = payload["M_matrix"]
    else:
        M_matrix = payload

    if not torch.is_tensor(M_matrix):
        M_matrix = torch.tensor(M_matrix, dtype=torch.float32)

    return M_matrix.to(device=device, dtype=torch.float32)


def build_model(config, lead_tensor, M_matrix, cor_sources_tensor):
    return ori_ISTANet(
        L=lead_tensor,
        LayerNo=config["ISTANet_layers"],
        n_source=config["N_sources"],
        M_matrix=M_matrix,
        cor_sources=cor_sources_tensor,
        truned_channels=config["truned_num"],
        alpha_loss=config["alpha_loss"],
        beta_loss=config["beta_loss"],
        grad_step_alpha=config["grad_step_alpha"],
        Phi_1_dim=config["Phi_1_dim"],
        Phi_2_dim=config["Phi_2_dim"],
    )


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_parameter_count(num_params: int) -> str:
    if num_params >= 1_000_000:
        return f"{num_params:,} ({num_params / 1_000_000:.3f}M)"
    if num_params >= 1_000:
        return f"{num_params:,} ({num_params / 1_000:.3f}K)"
    return f"{num_params:,}"


def main():
    config = TRAIN_CONFIG

    set_global_seed(int(config["seed"]))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

    print("=" * 50)
    print("加载 Forward Model...")
    fwd = load_forward_fixed(config["fwd_path"])
    leadfield = np.array(fwd["sol"]["data"], dtype=np.float32, copy=True)
    print(f"Leadfield shape: {leadfield.shape}")
    print(f"Original Leadfield std: {leadfield.std():.6f}")

    leadfield = normalize_L(leadfield)
    print(f"Normalized Leadfield mean±std: {leadfield.mean():.6f} ± {leadfield.std():.6f}")

    print("=" * 50)
    print("加载数据集...")
    train_subset, test_subset = get_trainsim_testsim_dataset(
        config["data_path"],
        config["train_rate"],
        config["num_runs"],
        5,
        config["seed"],
        PreProcess_method=config["data_preprocess"],
    )
    print(f"训练集大小: {len(train_subset)}")
    print(f"测试集大小: {len(test_subset)}")

    cor_sources = fwd["source_rr"] * 1e3
    cor_sources = cor_sources.astype(np.float32)

    if config["n_channel"] is None:
        config["n_channel"] = leadfield.shape[0]
    print(f"N_channels: {config['n_channel']}")

    sample_source, _ = train_subset[0]
    if config["N_sources"] is None:
        config["N_sources"] = int(sample_source.shape[0])
    if config["timepoints"] is None:
        config["timepoints"] = int(sample_source.shape[1])
    print(f"N_sources: {config['N_sources']}, Timepoints: {config['timepoints']}")

    if config.get("use_swanlab", True):
        logger = swanlab.init(
            project=config["project"],
            workspace="BINE-ESI",
            experiment_name=config["exp_name"],
            config=config,
        )
    else:
        logger = _NullLogger()

    print("=" * 50)
    print("创建 DataLoader...")
    pin_memory = config["device"] == "cuda"
    persistent_workers = config["num_workers"] > 0
    seed = int(config["seed"])
    g = torch.Generator()
    g.manual_seed(seed)

    loader_kwargs = dict(
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    if persistent_workers:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_subset,
        batch_size=config["batch_size"],
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        **loader_kwargs,
    )

    lead_tensor = torch.from_numpy(leadfield).to(config["device"])
    cor_sources_tensor = torch.from_numpy(cor_sources).to(config["device"])

    print("=" * 50)
    print("读取 M_matrix...")
    M_matrix = load_M_matrix(config["m_matrix_path"], config["device"])
    if tuple(M_matrix.shape) != (config["N_sources"], config["N_sources"]):
        raise ValueError(
            f"M_matrix 维度不匹配，期望 {(config['N_sources'], config['N_sources'])}，"
            f"得到 {tuple(M_matrix.shape)}"
        )

    print("=" * 50)
    print("构建 ori_ISTANet 模型...")
    model = build_model(config, lead_tensor, M_matrix, cor_sources_tensor)
    trainable_params = count_trainable_parameters(model)
    total_params = count_total_parameters(model)
    config["trainable_params"] = trainable_params
    config["total_params"] = total_params
    print(f"模型可训练参数量: {format_parameter_count(trainable_params)}")
    print(f"模型总参数量: {format_parameter_count(total_params)}")
    if config.get("use_swanlab", True):
        logger.log({
            "model/trainable_params": trainable_params,
            "model/total_params": total_params,
        })

    print("=" * 50)
    print("初始化 Trainer...")
    trainer = Trainer(
        model=model,
        lead_tensor=lead_tensor,
        cor_sources_tensor=cor_sources_tensor,
        train_loaders=train_loader,
        sample_loaders=test_loader,
        eval_count_source_power_greater_than=config["eval_count_source_power_greater_than"],
        train_lr=config["lr"],
        train_num_epochs=config["train_num_epochs"],
        use_lr_scheduler=config["use_lr_scheduler"],
        lr_warmup_epochs=config["lr_warmup_epochs"],
        weight_decay=config["weight_decay"],
        grad_clip_norm=config["grad_clip_norm"],
        results_folder=config["results_folder"],
        device=config["device"],
        dtype=config["dtype"],
        logger=logger,
        eval_metric=config["eval_metric"],
        eval_metric_mode=config["eval_metric_mode"],
    )

    resume_milestone = None
    if config.get("resume", False) and config.get("resume_checkpoint_path"):
        checkpoint_path = config["resume_checkpoint_path"]
        if os.path.isfile(checkpoint_path):
            print(f"从完整路径恢复训练: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=config["device"])
            trainer.model.load_state_dict(checkpoint['model'])
            trainer.opt.load_state_dict(checkpoint['opt'])
            trainer.epoch = checkpoint.get('epoch', 0)
            trainer.best_preview_metric = checkpoint.get('best_preview_metric', None)
            resume_milestone = "__loaded__"
            print(f"  已加载 epoch: {trainer.epoch}, best_metric: {trainer.best_preview_metric}")
        else:
            resume_milestone = checkpoint_path
            print(f"从 milestone 恢复训练: {resume_milestone}")

    print("=" * 50)
    history = trainer.train(
        num_epochs=config["train_num_epochs"],
        save_and_sample_every_epoch=config["save_and_sample_every_epoch"],
        early_stop_patience=config.get("early_stop_patience", None),
        resume_milestone=resume_milestone if resume_milestone != "__loaded__" else None,
    )

    print("=" * 50)
    print("ori_ISTANet 训练完成!")
    print(f"训练损失历史: {len(history['train_loss'])} epochs")
    if history['best_metric'] is not None:
        print(f"最佳 {config['eval_metric']}: {history['best_metric']:.6f} (Epoch {history['best_epoch']})")


if __name__ == "__main__":
    main()
