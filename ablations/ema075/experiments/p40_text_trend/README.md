# P40 文本趋势实验准备说明

> **2026-09-22修复状态：已同步服务器，8项CPU回归通过；未启动训练，GPU实跑待验证。** 以[修复完成与监控说明](修复完成与监控说明.md)为最新状态；[最终准备审计](最终准备审计_20260922.md)保留修复前证据。

实验目的与结论边界以 [`实验目的与对照设计.md`](实验目的与对照设计.md) 为准。

## 实验设计

三组实验只改变算法或奖励函数，其余训练参数、模型、数据和随机种子保持一致：

三组训练的实验随机种子固定为 **45**。现有数据子集是在种子 42 下抽取后冻结的；三组共用完全相同的 Parquet 文件，因此这不会破坏算法对照。这里要区分“数据抽样种子 42”和“训练种子 45”。

启动器把45写入数据加载器、Python哈希种子和实验环境。各rank用相同种子初始化LoRA；生成阶段使用45+rank。CUDA仍可能存在非确定性。

| 运行名 | 优势估计 | 奖励函数 | 用途 |
|---|---|---|---|
| `grpo_reward_a` | GRPO | Reward A | 与 `harpo_reward_a` 比较资源和趋势 |
| `harpo_reward_a` | HARPO | Reward A | 控制奖励函数后的算法对照 |
| `harpo_reward_b` | HARPO | Reward B | HARPO 内部的奖励敏感性对照 |

## 固定资源路径

- 代码：`/data/codefile/anonymous/human_behavior_atlas`
- Conda 环境：`/data/conda/users/maviuseranonymous/envs/anonymous_harpo_p40`
- 共享模型，只读：`/share/155/pretrained/Qwen2.5-Omni-3B`
- 共享奖励模型，只读：`/share/155/pretrained/mashang/SAGEPO/reward_models/all-MiniLM-L6-v2`
- 文本子集：`/data/datasets/human_behavior_atlas-data/anonymous_text_trend`
- 输出根目录：`/data/codefile/anonymous/harpo_p40_runs/experiments`

文本子集使用随机种子 42。训练集包含 128 条，验证集包含 32 条；任务为
`anxiety_cls` 和 `depression_cls`，两个任务的样本数量相等，但阳性/阴性类别不均衡。验证集共32条中只有3条阳性。媒体列均为空。

## P40 兼容路线

Tesla P40 的计算能力是 6.1，不能直接使用项目默认的 vLLM、FlashAttention 和
BF16 路线。本目录采用以下替代：

- `actor_rollout_ref.rollout.name=hf`；
- eager attention；
- FP16冻结基座、FP32可训练LoRA、分层FSDP及FP16 autocast与ShardedGradScaler（GPU初始化排错后的设置，详见启动记录）；
- LoRA rank 16；
- 最多两张显式指定的 GPU。

这条路线用于小规模文本趋势实验。它不构成论文多模态性能、吞吐量或完整训练的复现。

## 安全边界

`run_experiment.py` 默认只打印命令。真正启动需要同时提供：

- `--execute`；
- `--registration-confirmed`。

启动前还必须重新检查 `nvidia-smi`、确认空闲 GPU，并完成资源登记。不得沿用文档示例中的
GPU 编号。共享模型和共享奖励模型只读使用，checkpoint 和日志只能写到实名输出目录。

## 文件作用

- `prepare_text_subset.py`：检查共享 HBA 模态并抽取确定性的纯文本子集。
- `verify_setup.py`：CPU 环境、共享模型、数据和算法注册预检。
- `apply_p40_compat.py`、`apply_training_fixes.py`：给个人verl副本应用兼容修复、种子和记录/评估补丁。
- `text_dataset.py`：实际训练输入，统一512 token中间截断。
- `run_experiment.py`：生成或执行三组统一参数的实验命令。
- `resource_monitor.py`：自动包住训练进程，记录墙钟时间、GPU 利用率、峰值显存、估算 GPU 能耗，并把控制台输出保存为 `training.log`。
- `verify_run_complete.py`：训练结束后检查退出码、训练种子、最终 checkpoint 和最终测试产物。
- `reward_a_local.py`、`reward_b_local.py`：使用共享本地嵌入模型的两种奖励函数。

## 当前验证边界

训练器已配置为最后一步保存并重新加载checkpoint，再进行固定验证集贪心解码评估。结果在`final_test/`，该目录名不意味着独立测试集。启动器自动记录manifest、资源CSV/summary、training.log和steps.jsonl，通过严格验收才写complete。

8项CPU回归已通过，三组Hydra配置解析通过。未加载GPU模型权重或启动训练；每次总训练步数仍须显式指定。使用`--detach`可断开SSH，CPU断线存活检查已通过。启动初期每60秒、稳定后每15分钟查看，资源采样每5秒自动进行。
