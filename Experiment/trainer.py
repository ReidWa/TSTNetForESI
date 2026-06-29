import sys
from pathlib import Path
import math
nb_dir = Path().resolve()
parent_dir = nb_dir.parent
sys.path.insert(0, str(parent_dir))

import inspect

import torch
import numpy as np
from tqdm.auto import tqdm

import os
from .evaluator_torch import EvaluatorTorch as Evaluator


class Trainer:
    def __init__(self,
                 model,
                 lead_tensor, 
                 cor_sources_tensor,
                 train_loaders,
                 sample_loaders,
                 eval_count_source_power_greater_than = 0.1,
                 train_lr=1e-4,
                 train_num_epochs: int = 100,
                 use_lr_scheduler: bool = True,
                 lr_warmup_epochs: int = 0,
                 adam_betas=(0.9, 0.99),
                 weight_decay: float = 1e-4,
                 grad_clip_norm: float | None = None,
                 results_folder='./results/sample',
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 dtype=torch.float32,
                 logger=None,
                 eval_metric: str = 'SD',
                 eval_metric_mode: str = 'min',
                 log_grad_stats: bool = False):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model = model.to(device=device, dtype=dtype)
        # 避免每个 batch 都 inspect.signature（开销大）
        _sig = inspect.signature(self.model.compute_loss)
        self._loss_needs_physics = (
            'noise_true' in _sig.parameters and 'sparsity_true' in _sig.parameters
        )
        self.log_grad_stats = log_grad_stats
        self._cuda_non_blocking = device == 'cuda'
        self.train_loaders = train_loaders
        self.sample_loaders = sample_loaders
        self.evaluator = Evaluator(
            lead_tensor=lead_tensor,
            cor_sources_tensor=cor_sources_tensor,
            count_source_power_greater_than=eval_count_source_power_greater_than)
        self.logger = logger
        
        # 训练参数
        self.base_lr = train_lr
        self.train_num_epochs = train_num_epochs
        self.use_lr_scheduler = use_lr_scheduler  # 新增：是否使用学习率调度
        self.lr_warmup_epochs = lr_warmup_epochs
        self.grad_clip_norm = grad_clip_norm
        self.results_folder = results_folder
        self.eval_metric = eval_metric
        self.eval_metric_mode = eval_metric_mode

        
        # 优化器（CUDA 上优先 fused AdamW，通常更快）
        _aw_kw = dict(
            params=self.model.parameters(),
            lr=train_lr,
            betas=adam_betas,
            weight_decay=weight_decay,
        )
        if device == 'cuda':
            try:
                self.opt = torch.optim.AdamW(fused=True, **_aw_kw)
            except (TypeError, ValueError):
                self.opt = torch.optim.AdamW(**_aw_kw)
        else:
            self.opt = torch.optim.AdamW(**_aw_kw)
        
        # 训练状态
        self.epoch = 0
        self.best_preview_metric = None
        
        # 创建保存目录
        os.makedirs(results_folder, exist_ok=True)
    
    def train_one_epoch(self, epoch_idx: int = 0):
        """
        训练一个epoch
        
        Args:
            epoch_idx: 当前epoch的索引
            
        Returns:
            dict: 包含平均损失等训练指标的字典
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        # 用于累积所有要记录的指标
        accumulated_metrics = {}
        
        # 使用tqdm显示进度条
        pbar = tqdm(
            self.train_loaders,
            desc=f'Epoch {epoch_idx}',
            total=len(self.train_loaders)
        )
        
        nb = self._cuda_non_blocking
        for batch_idx, batch in enumerate(pbar):
            # 解包 batch:
            #   (source, eeg) 或 (source, eeg, noise_true, sparsity_true)
            if self._loss_needs_physics:
                if not (isinstance(batch, (list, tuple)) and len(batch) == 4):
                    raise TypeError(
                        '当前模型的 compute_loss 需要 noise_true / sparsity_true，'
                        '请让 Dataset 返回 (source, eeg, noise_true, sparsity_true)，'
                        '例如各 run 目录下提供 labels.npz 并扩展 datasetsim_singlerun。'
                    )
                source, eeg, noise_true, sparsity_true = batch
                source = source.to(
                    device=self.device, dtype=self.dtype, non_blocking=nb)
                eeg = eeg.to(device=self.device, dtype=self.dtype, non_blocking=nb)
                noise_true = noise_true.to(
                    device=self.device, dtype=self.dtype, non_blocking=nb)
                sparsity_true = sparsity_true.to(
                    device=self.device, dtype=self.dtype, non_blocking=nb)
            else:
                source, eeg = batch
                source = source.to(
                    device=self.device, dtype=self.dtype, non_blocking=nb)
                eeg = eeg.to(device=self.device, dtype=self.dtype, non_blocking=nb)
            
            # 清零梯度
            self.opt.zero_grad(set_to_none=True)
            
            # 前向传播，计算损失
            if self._loss_needs_physics:
                loss, log_dict = self.model.compute_loss(
                    eeg, source, noise_true, sparsity_true)
            else:
                loss, log_dict = self.model.compute_loss(eeg, source)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm
                )
            
            # 梯度全量统计非常耗时；默认关闭，需要时 Trainer(..., log_grad_stats=True)
            grad_stats = (
                self._compute_gradient_stats()
                if self.log_grad_stats
                else {}
            )
            
            # 更新参数
            self.opt.step()
            
            # 累积损失用于记录
            total_loss += loss.item()
            num_batches += 1
            
            # 累积 log_dict 中的指标
            for key, value in log_dict.items():
                if key not in accumulated_metrics:
                    accumulated_metrics[key] = 0.0
                # 支持 tensor 和 float
                if torch.is_tensor(value):
                    accumulated_metrics[key] += value.item()
                else:
                    accumulated_metrics[key] += value
            
            # 累积梯度统计
            for key, value in grad_stats.items():
                if key not in accumulated_metrics:
                    accumulated_metrics[key] = 0.0
                accumulated_metrics[key] += value
            
            # 获取当前学习率
            current_lr = self.opt.param_groups[0]['lr']
            
            # 更新进度条显示
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        # 计算平均损失和平均指标
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_metrics = {key: value / num_batches for key, value in accumulated_metrics.items()}
        
        # 记录到 logger
        if self.logger is not None:
            log_data = {'train/loss': avg_loss, 'train/lr': self.opt.param_groups[0]['lr']}
            for key, value in avg_metrics.items():
                log_data[f'train/{key}'] = value
            self.logger.log(log_data, step=epoch_idx)
        
        # 返回epoch的统计信息
        epoch_metrics = {
            'epoch': epoch_idx,
            'avg_loss': avg_loss,
            'num_batches': num_batches
        }
        
        # 更新当前epoch
        self.epoch = epoch_idx + 1
        
        return epoch_metrics
    
    def train(self, 
              num_epochs: int | None = None,
              save_and_sample_every_epoch: int = 10,
              early_stop_patience: int | None = None,
              resume_milestone: str | None = None):
        """
        完整的训练方法，使用epoch数量作为控制
        
        Args:
            num_epochs: 总训练epoch数，None则使用初始化时设置的train_num_epochs
            save_and_sample_every_epoch: 每隔多少个epoch保存模型并进行采样评估
            early_stop_patience: 早停耐心值，None表示不使用早停
            resume_milestone: 从指定的checkpoint恢复训练，None表示从头开始
            
        Returns:
            dict: 训练历史记录
        """
        # 使用初始化时的epoch数，或者覆盖
        if num_epochs is None:
            num_epochs = self.train_num_epochs
        
        # 恢复训练（如果指定）
        start_epoch = self.epoch  # 默认使用当前 epoch（支持外部预加载 checkpoint）
        if resume_milestone is not None:
            self.load(resume_milestone)
            start_epoch = self.epoch
            print(f"从checkpoint {resume_milestone} 恢复训练，当前epoch: {self.epoch}")
        elif self.epoch > 0:
            print(f"检测到预加载的 checkpoint，从 epoch {self.epoch} 继续训练")
        
        # 训练历史记录
        history = {
            'train_loss': [],
            'eval_metrics': [],
            'best_metric': self.best_preview_metric,
            'best_epoch': -1
        }
        
        # 早停计数器
        patience_counter = 0
        
        print(f"开始训练: epoch {start_epoch} -> {num_epochs}")
        print(f"设备: {self.device}, 数据类型: {self.dtype}")
        if self.use_lr_scheduler:
            print(f"学习率: {self.base_lr} (Warmup {self.lr_warmup_epochs} epochs + Cosine Decay)")
        else:
            print(f"学习率: {self.base_lr} (固定)")
        print("-" * 50)
        
        for epoch in range(start_epoch, num_epochs):
            # 调整学习率（在epoch开始时）
            current_lr = self._adjust_lr()
            
            # 训练一个epoch
            epoch_metrics = self.train_one_epoch(epoch_idx=epoch)
            history['train_loss'].append(epoch_metrics['avg_loss'])
            
            print(f"\nEpoch {epoch}/{num_epochs-1} 完成 | "
                  f"平均损失: {epoch_metrics['avg_loss']:.6f} | "
                  f"学习率: {current_lr:.2e}")
            
            # 定期采样评估（不再保存每个checkpoint，只在评估时更新best模型）
            if save_and_sample_every_epoch > 0 and (epoch + 1) % save_and_sample_every_epoch == 0:
                # 采样评估
                eval_result = self._sample_and_evaluate(epoch)
                history['eval_metrics'].append({
                    'epoch': epoch,
                    'metrics': eval_result
                })
                
                # 检查是否是最佳模型
                current_metric = eval_result.get(self.eval_metric, None)
                if self._is_preview_better(current_metric):
                    self.best_preview_metric = current_metric
                    history['best_metric'] = current_metric
                    history['best_epoch'] = epoch
                    patience_counter = 0
                    
                    # 保存最佳模型（记录当前评估指标）
                    self.save('best', eval_metrics=eval_result)
                    print(f"  ★ 新的最佳模型! {self.eval_metric}: {current_metric:.6f}")
                else:
                    patience_counter += 1
                    
                # 早停检查
                if early_stop_patience is not None and patience_counter >= early_stop_patience:
                    print(f"\n早停触发! 已经 {patience_counter} 个评估周期没有改进")
                    break
        
        # 评估并保存最终模型
        print(f"\n评估最终模型...")
        final_eval_result = self._sample_and_evaluate(self.epoch - 1)
        final_metric = final_eval_result.get(self.eval_metric, None)
        self.save('final', eval_metrics=final_eval_result)
        
        print("-" * 50)
        print(f"训练完成! 最终模型已保存")
        print(f"最终epoch: {self.epoch}")
        if final_metric is not None:
            print(f"最终模型 {self.eval_metric}: {final_metric:.6f}")
        if history['best_metric'] is not None:
            print(f"最佳 {self.eval_metric}: {history['best_metric']:.6f} (Epoch {history['best_epoch']})")
        
        return history
    
    def _sample_and_evaluate(self, epoch: int) -> dict:
        """
        使用sample_loaders进行采样并评估
        
        Args:
            epoch: 当前epoch索引
            
        Returns:
            dict: 评估指标字典
        """
        self.model.eval()
        
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(
                self.sample_loaders, 
                desc=f'Epoch {epoch} 评估',
                leave=False
            )):
                # 解包 batch: (source, eeg) 或 (source, eeg, noise_true, sparsity_true)
                if isinstance(batch, (list, tuple)) and len(batch) == 4:
                    source, eeg, _, _ = batch
                else:
                    source, eeg = batch
                nb = self._cuda_non_blocking
                source = source.to(
                    device=self.device, dtype=self.dtype, non_blocking=nb)
                eeg = eeg.to(device=self.device, dtype=self.dtype, non_blocking=nb)
                
                # 使用模型进行采样，eeg 作为条件输入
                samples = self.model.predict(eeg)
                # 计算评估指标: 
                # - pred: 预测的源信号
                # - true_cor: 源坐标 (来自 self.evaluator 内部的 cor_sources_tensor)
                # - true_sig: 真实源信号
                metrics = self.evaluator.evaluate(pred=samples, true_cor=None, true_sig=source)
                all_metrics.append(metrics)
        
        # 聚合所有batch的指标
        aggregated_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                aggregated_metrics[key] = np.mean(values)
        
        # 打印评估结果
        print(f"  评估结果:")
        for key, value in aggregated_metrics.items():
            print(f"    {key}: {value:.6f}")
        
        # 记录到logger
        if self.logger is not None:
            log_dict = {f'eval/{k}': v for k, v in aggregated_metrics.items()}
            self.logger.log(log_dict, step=epoch)
        
        self.model.train()
        return aggregated_metrics

    def save(self, milestone, eval_metrics: dict | None = None):
        data = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'best_preview_metric': self.best_preview_metric,
            'eval_metrics': eval_metrics,  # 保存当前模型的评估指标
        }
        torch.save(data, os.path.join(self.results_folder, f'model-{milestone}.pt'))

    def load(self, milestone):
        path = os.path.join(self.results_folder, f'model-{milestone}.pt')
        if os.path.exists(path):
            data = torch.load(str(path), map_location=self.device)
            self.model.load_state_dict(data['model'])
            self.epoch = data.get('epoch', 0)
            self.opt.load_state_dict(data['opt'])
            self.best_preview_metric = data.get('best_preview_metric', None)
            print(f"load model - {path}, epoch: {self.epoch}")

    def _adjust_lr(self):
        """
        学习率调整策略
        - use_lr_scheduler=True: Linear warmup + cosine decay (到 0.1 * base_lr)
        - use_lr_scheduler=False: 固定学习率
        """
        if not self.use_lr_scheduler:
            # 固定学习率模式：始终使用 base_lr
            lr = self.base_lr
        elif self.lr_warmup_epochs > 0 and self.epoch < self.lr_warmup_epochs:
            # Warmup阶段：线性增加学习率
            lr = self.base_lr * (self.epoch + 1) / self.lr_warmup_epochs
        else:
            # Cosine decay阶段
            progress = (self.epoch - self.lr_warmup_epochs) / max(1, self.train_num_epochs - self.lr_warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.base_lr * (0.1 + 0.9 * cosine)
        
        for g in self.opt.param_groups:
            g['lr'] = lr
        return lr

    def _is_preview_better(self, value: float | None) -> bool:
        if value is None or not np.isfinite(value):
            return False
        if self.best_preview_metric is None:
            return True
        if self.eval_metric_mode == 'min':
            return value < self.best_preview_metric
        return value > self.best_preview_metric

    def _compute_gradient_stats(self) -> dict:
        """
        计算模型梯度的统计信息
        
        Returns:
            dict: 包含梯度统计信息的字典
        """
        grad_stats = {}
        
        # 收集所有梯度
        all_grads = []
        layer_grad_norms = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                grad_flat = grad.view(-1)
                all_grads.append(grad_flat)
                
                # 计算每层的梯度范数
                layer_norm = grad.norm(2).item()
                # 简化层名称（取最后两级）
                short_name = '/'.join(name.split('.')[-2:]) if '.' in name else name
                layer_grad_norms[short_name] = layer_norm
        
        if all_grads:
            # 拼接所有梯度
            all_grads_tensor = torch.cat(all_grads)
            
            # 总体梯度统计
            grad_stats['grad/total_norm'] = all_grads_tensor.norm(2).item()
            grad_stats['grad/mean'] = all_grads_tensor.mean().item()
            grad_stats['grad/std'] = all_grads_tensor.std().item()
            grad_stats['grad/max'] = all_grads_tensor.max().item()
            grad_stats['grad/min'] = all_grads_tensor.min().item()
            
            # 记录每层梯度范数（可选：只记录前几层和后几层，避免日志过多）
            sorted_layers = sorted(layer_grad_norms.items(), key=lambda x: x[1], reverse=True)
            # 记录梯度最大的5层
            for i, (layer_name, norm) in enumerate(sorted_layers[:5]):
                grad_stats[f'grad_layer_top{i+1}/{layer_name}'] = norm
        
        return grad_stats