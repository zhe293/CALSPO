# CALSPO / EquiAffect-3B

**Class-Aware Supervised Policy Optimization（类别感知监督策略优化）**。

本仓库同时交付 **CALSPO训练算法**与使用该算法训练得到的 **EquiAffect-3B模型权重**，并保存训练与评估数据、两组 AV + HARPO 消融实现及论文初稿。代码基于 Human Behavior Atlas / OmniSapiens 和 VERL，具体 AV 优势恢复、HARPO 调制及类别加权标签监督实现见 `verl/verl/trainer/ppo/`。这是个人科研项目的私有归档，计划于论文终版投稿时转为公开。

## 三组配置

| 配置 | 任务尺度平滑：旧值／新值 | AV | 类别加权标签监督 |
|---|---|---|---|
| CALSPO | 75%／25% | 保留 | 保留，λ=1 |
| ema075 | 75%／25% | 保留 | 去掉 |
| ema095 | 95%／5% | 保留 | 去掉 |

三份实现分别来自对应实验使用的代码快照，关键文件与原始运行清单中的 SHA-256 一致。上述平滑比例特指 HARPO 任务尺度更新，不代表所有平滑参数。历史运行设置完整保存在各自 `provenance/source_run_manifest.json`；便携入口默认用相同随机种子，便于后续控制变量重跑。归档代码不代表消融结果已经全部完成。

## 文件布局

```text
verl/                         CALSPO 完整 VERL 实现
training/                     训练模板与原项目训练代码
evaluation/                   原项目评估代码
ablations/ema075/              去监督，75%／25% 的完整代码快照
ablations/ema095/              去监督，95%／5% 的完整代码快照
datasets/train.parquet        304 题训练集，便携媒体路径
datasets/validation.parquet   原实验使用的 216 题验证数据
datasets/eval328.parquet      冻结的 328 题评估集，便携媒体路径
datasets/media/               所需的 887 个音视频文件
datasets/original/            原始 Parquet 文件，逐字节保留
datasets/provenance/eval328/   评估集构建、筛选和核验记录
models/EquiAffect-3B/         训练后LoRA权重、配置、校验及加载说明
paper/                        同步的Markdown、LaTeX和PDF初稿
models/base_model_metadata/   基座模型配置、分词器、许可等
models/base_model_manifest.json  固定版本与权重校验信息
provenance/                   原始运行、环境及采集记录
scripts/                      便携运行、模型下载、文件核验入口
```

数据移植只替换媒体路径，题目、标签、顺序及表结构保持不变；原路径与便携路径对应关系见 `datasets/media_mapping.json`。媒体使用短文件名以兼容 Windows。328 题集沿用现有冻结数据及其原始构建记录，仓库不改写数据来源和选择过程。训练集含情绪分类 224 题、抑郁分类 80 题；评估集分别为 287 和 41 题。

## EquiAffect-3B训练后模型

EquiAffect-3B 以 Qwen2.5-Omni-3B 为基座，使用 CALSPO 联合训练情绪与抑郁分类。第38步检查点的完整训练后 LoRA 适配权重保存在 [models/EquiAffect-3B](models/EquiAffect-3B/README.md)，与固定版本的基座共同加载。权重使用 Git LFS 保存；克隆后运行 `git lfs pull` 获取实际权重文件，再运行 `python scripts/verify_equiaffect.py` 核验大小、SHA-256及目标层覆盖。

本次从完整 FSDP 检查点提取全部 LoRA 参数，修复了原导出文件漏掉语言解码层的问题；已核验参数覆盖、有限值和原导出参数的一致性。没有重新运行 GPU 训练或推理。加载说明与来源记录见模型目录。

论文：[Markdown初稿](paper/CALSPO论文中文初稿.md)、[LaTeX源文件](paper/latex/main.tex)、[PDF预览](paper/latex/main.pdf)。论文将 CALSPO 与 EquiAffect-3B 分别作为算法和模型产出，新增 Wang 等（Interspeech 2023）的多数中性类过拟合研究作为背景引用。

## 基座模型

实际实验基座为 **Qwen/Qwen2.5-Omni-3B**。GitHub 中包含配置、分词器、模型文件清单和下载入口。权重通过官方 Hugging Face 固定版本获取：

- 官方入口：https://huggingface.co/Qwen/Qwen2.5-Omni-3B
- 固定版本：`f75b40e3da2003cdd6e1829b1f420ca70797c34e`
- 许可：Qwen Research License，见 `models/base_model_metadata/LICENSE` 和 `NOTICE`。

```bash
pip install huggingface_hub
python scripts/download_base_model.py --output models/Qwen2.5-Omni-3B --embedding-output models/all-MiniLM-L6-v2
```

三个权重分片下载后自动核验大小与 SHA-256。奖励实现还使用本地 `all-MiniLM-L6-v2`；该辅助模型通过可选参数下载。

## 环境与运行

训练在 Linux / CUDA 环境运行。本地 Windows 可阅读、校验和打印运行参数。原实验环境为 Python 3.10、PyTorch 2.5.1+cu118、Transformers 4.52.2、Ray 2.44.1、PEFT 0.15.2；完整已观察版本见 `provenance/environment_snapshot.json`。该记录是原环境快照，并非新机器安装成功的证明。原项目安装文件见 `verl/requirements.txt`、`verl/setup.py` 和 `scripts/setup_verl.sh`；已归档的源码包含实验需要的改动，安装时使用本仓库而不是重新覆盖为上游版本。

```bash
bash scripts/setup_verl.sh
# 先在自己的 Linux 环境补齐与环境快照一致的依赖。
python scripts/verify_assets.py
python scripts/run_experiment.py --variant calspo --gpu 0 --base-model models/Qwen2.5-Omni-3B --embedding-model models/all-MiniLM-L6-v2 --output outputs/calspo
```

运行入口默认仅打印解析后的参数；添加 `--execute` 才启动训练。将 `--variant` 改为 `ema075` 或 `ema095` 即切换对应完整实现。每组使用独立空输出目录，禁用自动续训。原实验批大小 8、每题 4 个采样回答、学习率 1e-6、LoRA rank 16 / alpha 32、38 个训练步等设置从源清单继承。

媒体路径均相对Parquet所在的 `datasets/` 目录，形式为 `media/文件名`。核验脚本按实际加载器的目录规则检查全部983个媒体引用：训练296个、验证216个、328题评估471个；同时校验887个媒体文件与三份核心代码快照。训练和评估的 `--execute` 入口会先运行该核验，缺失、空文件或错误路径会中止启动。安装脚本直接安装已包含改动的源码，不初始化子模块或应用外部补丁；`bash scripts/setup_verl.sh --check-only` 可仅检查安装来源。

## 论文结果复现

[paper/README.md](paper/README.md)说明完整逐条回答、原始指标、图表源数据与复算流程。两种模型各328条完整输出按样本身份对齐。

```bash
python -m pip install -r paper/requirements.txt
python paper/recompute_metrics.py
python paper/plot_figures.py
```

复算与绘图均可在CPU上运行，绘图检查工具已随仓库附带，无需本机科研技能目录。图中CALSPO对应EquiAffect-3B。

在 328 题数据上评估已完成的 checkpoint：

```bash
python scripts/run_experiment.py --variant calspo --mode evaluate --gpu 0 --base-model models/Qwen2.5-Omni-3B --embedding-model models/all-MiniLM-L6-v2 --checkpoint outputs/calspo/global_step_38 --output outputs/calspo_eval328 --execute
```

运行前须按所在服务器规则确认 GPU 与登记。本次归档仅进行了 CPU 文件、数据和参数核验，没有重新运行 GPU 训练或评估。历史辅助脚本可能保留原服务器路径，建议使用 `scripts/run_experiment.py`。

## 来源与许可

原方法项目：[OmniSapiens / Human Behavior Atlas](https://github.com/MIT-MI/human_behavior_atlas)。VERL 的 Apache 2.0 许可保存在各代码快照的 `verl/LICENSE`。模型许可独立适用；原数据及原项目代码仍受各自来源条款约束，本仓库不新增统一的再授权许可。
