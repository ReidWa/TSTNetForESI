"""
从 datapath/run-XXXXX/ 目录提取 sim_run_eegs.npz 和 sim_source.npz 
复制到 savepath/run-XXXXX/，保持 run-id 一致
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== 配置 ====================
datapath = Path(r"E:\2_ESIdiff\Dataset\localize-mi-data\derivatives\simulationV2\trainset\collection_all\sub-01\runs_dataV2")  # 源数据路径，请根据实际情况修改
savepath = Path(r"E:\2_ESIdiff\Dataset\localize-mi-data\derivatives\simulationV2\trainset\collection_all\sub-01\forautodl")  # 目标保存路径，请根据实际情况修改

num_runs = 20000  # run 的总数
num_workers = 8   # 并行线程数

# 要提取的文件列表
files_to_extract = ["sim_run_eegs.npz", "sim_source.npz"]
# =============================================


def copy_run_files(run_id: int) -> tuple:
    """
    复制单个 run 的文件
    
    Args:
        run_id: run 编号
    
    Returns:
        (run_id, success, message)
    """
    run_name = f"run-{run_id:05d}"
    src_dir = datapath / run_name
    dst_dir = savepath / run_name
    
    # 检查源目录是否存在
    if not src_dir.exists():
        return (run_id, False, f"源目录不存在: {src_dir}")
    
    # 创建目标目录
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    missing_files = []
    
    for filename in files_to_extract:
        src_file = src_dir / filename
        dst_file = dst_dir / filename
        
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            copied_files.append(filename)
        else:
            missing_files.append(filename)
    
    if missing_files:
        return (run_id, False, f"缺少文件: {missing_files}")
    
    return (run_id, True, f"成功复制 {len(copied_files)} 个文件")


def main():
    print(f"源路径: {datapath}")
    print(f"目标路径: {savepath}")
    print(f"总 run 数: {num_runs}")
    print(f"要提取的文件: {files_to_extract}")
    print(f"并行线程数: {num_workers}")
    print("-" * 50)
    
    # 创建目标根目录
    savepath.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    failed_runs = []
    
    # 使用多线程加速复制
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(copy_run_files, i): i for i in range(1, num_runs + 1)}
        
        # 使用 tqdm 显示进度
        with tqdm(total=num_runs, desc="复制进度", unit="run") as pbar:
            for future in as_completed(futures):
                run_id, success, msg = future.result()
                
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    failed_runs.append((run_id, msg))
                
                pbar.update(1)
                pbar.set_postfix({"成功": success_count, "失败": fail_count})
    
    # 打印结果统计
    print("-" * 50)
    print(f"复制完成!")
    print(f"成功: {success_count} / {num_runs}")
    print(f"失败: {fail_count} / {num_runs}")
    
    # 如果有失败的，打印详情（最多显示20个）
    if failed_runs:
        print(f"\n失败的 run (显示前20个):")
        for run_id, msg in sorted(failed_runs)[:20]:
            print(f"  run-{run_id:05d}: {msg}")
        
        if len(failed_runs) > 20:
            print(f"  ... 还有 {len(failed_runs) - 20} 个失败")
        
        # 保存完整的失败列表到文件
        fail_log = savepath / "failed_runs.txt"
        with open(fail_log, "w", encoding="utf-8") as f:
            for run_id, msg in sorted(failed_runs):
                f.write(f"run-{run_id:05d}: {msg}\n")
        print(f"\n完整失败列表已保存到: {fail_log}")


if __name__ == "__main__":
    main()
