# TST-Net Paper Code Snapshot

本目录整理了论文 **Tikhonov Shrinkage-Threshold Network: A Single-Step Inverse Solver for EEG Source Imaging** 中与模拟数据生成、单被试模拟训练和独立模拟测试直接相关的代码快照。

整理日期：2026-06-28  

## 范围

已包含：

- `sub-01` 个体化模拟数据生成入口。
- 模拟数据收集、标签生成和皮层拓扑 `M_matrix` 构建的辅助脚本。
- 单被试模拟数据训练入口：TST-Net 和标准 ISTANet。
- 独立模拟测试集评估入口：TST-Net 和标准 ISTANet。
- 上述入口依赖的最小本地包：`data/`、`Experiment/`、`model/ISTANet/`、`utils/`。
- 环境文件：`requirements.txt`、`environment.yml`。

未包含：

- 真实 CCEP 数据预处理、真实数据测试和多被试训练调度脚本。
- 可视化、测速、SwanLab 导出、DeterminedAI/远程提交、历史实验日志。
- checkpoint、模拟数据、真实数据、结果表、图片和 notebook。
- 其他未用于本文主线的模型族，例如 diffusion、DeepSIF、ADMMNet、GBFs wrapper。

## 文件清单

### 模拟数据

- `00-2-sim_01_runs_data.py`  
  生成 `sub-01` 模拟训练集和 12 个独立模拟测试条件。默认输出到 BIDS 根目录下的 `derivatives/simulationV3/trainset` 和 `derivatives/simulationV3/testset`。

- `00-2-sim_02_collectsimdata.py`  
  收集分散的模拟 run 文件，便于后续训练入口统一读取。

- `00-2-sim_03collectsimdatawithprepro.py`  
  收集带预处理版本的模拟数据。

- `00-2-sim_03_generate_labels.py`  
  从模拟源活动中生成 trial-level 标签和统计信息。

- `00-2-sim_06_build_M_matrix.py`  
  从 forward solution 的皮层源空间构建 `M_matrix.pt`，供 ISTANet/TST-Net 的空间项使用。

### 训练

- `01-Train-main_oriISTAwithSVDGrad.py`  
  TST-Net 训练入口。核心模型为 `model/ISTANet/ori_ISTANetwithSVDgrad.py`。

- `01-Train-main_oriISTA.py`  
  标准 ISTANet 训练入口。核心模型为 `model/ISTANet/ori_ISTANet.py`。

### 独立模拟测试

- `02-Test-TSTNet_on_simtest_sub01.py`  
  加载 TST-Net checkpoint，在 `sub-01` 独立模拟测试集上计算 SD、DLE、AUC、SE，并导出 trial-level 和 summary CSV。

- `02-Test-oriISTA_on_simtest_sub01.py`  
  加载 ISTANet checkpoint，在同一模拟测试集上计算 SD、DLE、AUC、SE。

## 运行前需要修改的配置

这些脚本保留了原始代码库中的绝对路径和实验配置。运行前至少检查：

- `00-2-sim_01_runs_data.py`
  - `--dir-bids`
  - `--subj`
  - `--run`
  - `--train-total`
  - `--train-chunk-size`
  - `--train-trials`

- `00-2-sim_06_build_M_matrix.py`
  - `BUILD_M_CONFIG["fwd_path"]`
  - `BUILD_M_CONFIG["save_path"]`
  - `sigma_mm`
  - `limit_mm`

- `01-Train-main_oriISTAwithSVDGrad.py` 和 `01-Train-main_oriISTA.py`
  - `TRAIN_CONFIG["data_path"]`
  - `TRAIN_CONFIG["fwd_path"]`
  - `TRAIN_CONFIG["m_matrix_path"]`
  - `TRAIN_CONFIG["results_folder"]`
  - `TRAIN_CONFIG["use_swanlab"]`
  - `ISTANet_layers`
  - TST-Net 的 `truned_num`、`Phi_1_dim`、`Phi_2_dim`

- `02-Test-TSTNet_on_simtest_sub01.py` 和 `02-Test-oriISTA_on_simtest_sub01.py`
  - `CFG["dir_bids"]`
  - `CFG["test_data_path"]`
  - `CFG["m_matrix_path"]`
  - `CFG["checkpoint_path"]`
  - `CFG["save_dir"]`
  - 模型层数、rank 和 `Phi` 容量必须与 checkpoint 的 `state_dict` 形状一致。

注意：历史脚本、结果目录和部分 CSV 中可能存在 `TSTNet-low`、`TSTNet-Lite`、`TSTNet-Medium`、`TSTNet-High` 标签错写。复现实验时以 checkpoint `state_dict` 形状、脚本中的 `truned_num`、`Phi_1_dim`、`Phi_2_dim` 和结果顶层目录交叉核对，不要只相信文件名或 CSV 内部 `method` 字段。

## 基本流程

以下命令假设当前目录为 `paper_code`。

### 1. 安装环境

```bash
conda env create -f environment.yml
conda activate esidiff
pip install -r requirements.txt
```

如已有可用的 MNE/PyTorch 环境，可只补装缺失依赖。

### 2. 生成 sub-01 模拟训练和测试数据

```bash
python 00-2-sim_01_runs_data.py ^
  --dir-bids E:\2_ESIdiff\Dataset\localize-mi-data ^
  --subj sub-01 ^
  --run run-01 ^
  --train-total 20000 ^
  --train-chunk-size 2000 ^
  --train-trials 20 ^
  --max-workers 12 ^
  --disable-figs
```

该入口会生成：

- 训练集：随机源数、扩散范围、SNR 和波形参数。
- 测试集：`firingsource_1..4`、`snr_minus5dB/0dB/5dB/10dB`、`extents_0mm/10mm/20mm/30mm`。

### 3. 构建 M_matrix

先在 `00-2-sim_06_build_M_matrix.py` 中改好 `fwd_path` 和 `save_path`，再运行：

```bash
python 00-2-sim_06_build_M_matrix.py
```

如果已有经过核对的 `M_matrix.pt`，可直接在训练和测试脚本中指向该文件。

### 4. 训练 TST-Net 或 ISTANet

先修改训练脚本顶部的 `TRAIN_CONFIG`。TST-Net 示例：

```bash
python 01-Train-main_oriISTAwithSVDGrad.py
```

ISTANet 示例：

```bash
python 01-Train-main_oriISTA.py
```

训练入口会在 `results_folder` 下保存 `model-best.pt` 和 `model-final.pt`。

### 5. 在独立模拟测试集上评估

先修改测试脚本顶部的 `CFG`，尤其是 checkpoint 和输出目录。TST-Net 示例：

```bash
python 02-Test-TSTNet_on_simtest_sub01.py
```

ISTANet 示例：

```bash
python 02-Test-oriISTA_on_simtest_sub01.py
```

测试脚本会导出每个测试条件的 summary、trial-level 指标、参数分组统计和总体结果，指标包括 SD、DLE、AUC、SE。

## 代码依赖关系

- `data/dataPreprocessor.py`：模拟源和 EEG 生成。
- `data/sigGen.py`：源时间波形生成。
- `data/dataset.py`：模拟 run 读取和 PyTorch Dataset。
- `Experiment/trainer.py`：训练循环、验证和 checkpoint 保存。
- `Experiment/evaluator_torch.py`：SD、DLE、AUC、SE 指标计算。
- `model/ISTANet/ori_ISTANetwithSVDgrad.py`：TST-Net 的 Tikhonov-TSVD 单步 forward update 和收缩阈值模块。
- `model/ISTANet/ori_ISTANet.py`：标准 ISTANet baseline。
- `utils/simtest_result_export.py`：模拟测试结果导出。
- `utils/utils_train.py`、`utils/util_sim.py`、`utils/fx_bids_local_mi.py`：forward solution、BIDS 和模拟工具函数。

## 复现口径

论文当前主模拟实验口径为：

- 被试：`sub-01`。
- 训练：个体化导联场生成的模拟训练数据。
- 测试：相互独立的 `sub-01` 模拟测试集。
- 测试条件：4 个激活源数量条件、4 个 SNR 条件、4 个源扩展范围条件。
- 模拟指标：SD、DLE、AUC、SE。
- 结果汇总默认报告 `mean ± SD`，并明确 SD 的统计层级。

本目录只保存代码快照，不保存数据、权重或结果。复现实验时应记录所用数据路径、checkpoint 路径、模型容量、rank、随机种子和导联场文件版本。
