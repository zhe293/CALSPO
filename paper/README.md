# 论文结果复算与绘图

`results/HARPO/` 和 `results/CALSPO/` 分别保存论文对应第38步模型的328条完整输出、带样本来源的逐条预测、原始指标和详细指标。CALSPO结果对应EquiAffect-3B。按 `diagnostic_index` 与评估集原始顺序对齐；预测JSON的 `index` 是评估加载顺序，不能直接作为Parquet行号。

`results/source_manifest.json` 记录原文件SHA-256。`results/reported_metrics.json` 保存论文原有指标作为比对目标；`source_data/verified_data.json` 是从完整回答重新计算的指标与逐条结果。CSV文件提供图1–3、逐类指标和任务指标的源数据。

从仓库根目录运行（也可使用脚本的绝对路径从其他目录运行）：

```bash
python -m pip install -r paper/requirements.txt
python paper/recompute_metrics.py
python paper/plot_figures.py
```

复算程序使用原评估器的答案提取顺序及固定合法标签，验证两种模型各328条输出、样本对应关系、全部任务和模态指标，以及混淆矩阵与论文报告值的一致性。无效输出保留在错误分母中，中性文本前缀统计不改变正确率。该流程复算已归档输出，不重新执行模型推理。

绘图代码使用仓库内的 `plotting_qa/`，无需本机安装科研技能或存在`E:/科研`目录。它导出三组SVG、PDF、600dpi PNG及源数据CSV，并检查面板对齐、字体和图中文字碰撞。结果限于所报告评估集；原始来源记录位于仓库的 `datasets/provenance/eval328/`。
