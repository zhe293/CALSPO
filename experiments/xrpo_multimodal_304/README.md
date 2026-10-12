> 2026-09-24 Astra 最终修订：实际生效方案以 [Astra最终检查_20260924.md](Astra最终检查_20260924.md) 为准。下文历史方案中的独立 paper-style XRPO、任务×模态、按模态训练分批及同时等待两张卡等描述已被替换。

# 304 条多模态 XRPO 趋势实验

## 目的与比较

在 P40 硬件约束下，用相同基座模型、数据、奖励 B、随机种子和训练参数，对比原仓库 HARPO 与 XRPO 改动 2、3。观察验证集分类效果、训练稳定性、耗时和资源消耗。304 条训练数据与 64 条验证数据只支持探索性趋势，不能据此声称严格复现论文或统计显著。

计划的两组正式作业：

| 运行名 | 算法 | GPU | 改动 |
| --- | --- | --- | --- |
| `original_harpo` | 隔离副本中的原仓库 HARPO 优势实现 | 0 | 核心公式保留，任务分组统一为任务×模态 |
| `xrpo_23` | 独立的研究实现 | 2 | 惯性新观测权重 5%→10%；新增任务参照系数 `x_m = R_m / GM(R_m)` |

`paper_harpo` 也有配置，但本次自动启动只安排上表两组。改动 1（问题层幂次收缩）暂不执行。两组种子均为 45，每组 38 步，训练批次 8，每个问题生成 4 个回答，响应上限 256 token。两组均使用任务×模态作为优势统计分组，奖励函数仍收到原始任务名。

## 数据

数据目录：`/data/datasets/human_behavior_atlas-data/anonymous_multimodal_304_v4`。从共享数据集只读抽取，媒体文件保存在个人目录；训练与验证提示词没有重复。v4 保留 v3 的所有 304+64 条样本、标签及媒体，只纠正 80+16 条纯文本抑郁分类题目中错误要求依据录音语气和韵律判断的句子；新题目明确只依据书面转写。

| 任务/模态 | 训练 | 验证 |
| --- | ---: | ---: |
| 抑郁分类/文本 | 80 | 16 |
| 情绪分类/文本+音频 | 152 | 32 |
| 情绪分类/文本+视频+音频 | 72 | 16 |

选取的最长音频约 13.14 秒。处理器上限 15 秒，因此不会裁剪这些样本；视频均从原 MP4 抽取首尾两帧，再交给 Qwen 处理器。与原论文完整数据和多样任务不同，本文的验证主要反映这个受限子集。情绪类别不均衡，除准确率外还记录宏 F1、平衡准确率和无效输出率。

## 奖励 B

`reward_b_local.py` 调用作者的 `human_behaviour_harpo.py`。两组使用相同的奖励；分类标签、格式与长度惩罚均沿用作者实现。此实验把软长度惩罚的界限设为 256 token、缓冲区 64 token，并让奖励管理器传入分词器测得的输出长度。这样长度惩罚与实际生成上限一致。此前文本趋势实验的奖励参数与计长方式未必相同，不要直接用它们的分数与本实验对比。

## 代码与备份

- 原服务器仓库：`/data/codefile/anonymous/human_behavior_atlas`，未改动。
- 隔离实验仓库：`/data/codefile/anonymous/human_behavior_atlas_xrpo_304`。
- 本地原始文件备份：本目录 `server_baseline/`；本地改后副本：`patched/`。
- 共享模型和共享源数据均只读使用。
- 隔离仓库的 `rl_dataset.py`、`audio_utils.py`、`vision_utils.py`、`hf_rollout.py`、`batch.py` 负责真实媒体输入、生成和准确计长；`ray_trainer.py` 选择优势算法与记录日志；`xrpo_advantage.py` 实现研究中的两个自定义优势模式。

## 自动启动条件与查看方式

等待程序：`experiments/xrpo_multimodal_304/launch_when_available.py`。只有旧的 HARPO-B/GRPO-B 两组 `status.json` 都报告 `complete`，且 GPU0 与 GPU2 显存、计算进程检查均空闲，才会启动新两组。新作业前运行器会再次核对空闲。用户已确认新增对照实验获批且资源登记已更新。等待程序每 30 秒检查一次；它不预占 GPU。

启动前核查已补强：训练器把 `multi_modal_inputs` 与模态标识送入生成批次，并保留给 actor 计算；HF rollout 为 Qwen 的三轴位置编号追加回答位置；CPU 替身测试确认音频、视频特征确实传到模型的 `generate()` 参数。实际采样器验证了 38 个训练批次和 8 个验证批次，全部行恰好出现一次，每批只有一种模态。等待程序启动时锁定关键代码和数据文件的 SHA256，真正启动前再次比较，文件变化则停止提交。

新输出目录：`/data/codefile/anonymous/xrpo_multimodal_304_results_20260924`。

服务器上查看等待或运行状态：

```bash
cat /data/codefile/anonymous/xrpo_multimodal_304_results_20260924/watcher_status.json
tail -f /data/codefile/anonymous/xrpo_multimodal_304_results_20260924/watcher.log
tail -f /data/codefile/anonymous/xrpo_multimodal_304_results_20260924/launcher_original_harpo.log
tail -f /data/codefile/anonymous/xrpo_multimodal_304_results_20260924/launcher_xrpo_23.log
```

实际训练开始后，各运行目录有 `training.log`、`steps.jsonl`、`resource_usage.csv`、`resource_usage.summary.json`、最终验证结果和 `status.json`。若任何前置条件或训练失败，等待程序会写 `failed`，不会擅自重跑。

## 验证与边界

已完成：完整 304+64 条媒体处理器 CPU 检查、原仓库 HARPO 与研究优势函数的 CPU 算例、奖励 B 长度惩罚算例、两组 Hydra 配置解析。GPU 上完整训练和评测仍须实际完成，结果尚未知。GPU0 与 GPU2 虽同为 P40，但可用显存、同时运行的系统负载可能不同；计算资源比较应报告 GPU 小时、采样峰值显存、功耗、训练时间与吞吐量，并注明并行干扰。

`xrpo_advantage.py` 为研究实现，不等同于论文逐字实现；按任务×模态分批时，跨任务参照采用历史已见分组的平滑值。单轮训练通常不会再次看到同一问题，因而问题层的历史平滑效应有限。任何性能差异都不能单独归因于某个公式项。
