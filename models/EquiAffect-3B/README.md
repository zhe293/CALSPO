# EquiAffect-3B

EquiAffect-3B 是以 **CALSPO（Class-Aware Supervised Policy Optimization）** 训练得到的多模态行为分类模型。基座是 Qwen/Qwen2.5-Omni-3B，联合训练任务为七类情绪分类与二类抑郁分类。模型沿用基座架构。

## 交付内容

- `adapter_model.safetensors`：第38步检查点的完整训练后 LoRA 权重，164,459,640 字节（约164.46MB），使用 Git LFS 保存。
- `adapter_config.json`：实验导出的原始 PEFT 配置，rank 16、alpha 32。
- `source_manifest.json`：来源检查点、固定基座版本、文件大小与 SHA-256，以及权重结构核验结果。
- `extraction_manifest.json`：从完整 FSDP 模型状态提取适配权重的记录。

这是训练后的适配权重。完整 EquiAffect-3B 由固定版本的基座与该适配器共同组成。基座权重通过仓库的官方下载入口获取；本目录不包含优化器或完整 FSDP 训练状态。适配器配置保留原实验中的服务器路径，加载时请显式传入自己下载的基座目录。

原检查点单独导出的适配器只包含音频与视觉模块，漏掉语言解码层。本次从同一检查点的完整模型状态重新提取888个LoRA张量，覆盖444个目标模块；原导出的384个张量与重新提取后的对应张量逐字节一致。所有张量均为有限float32值。原始训练状态未修改。

克隆仓库后执行 `git lfs pull` 获取实际权重文件；GitHub网页中的LFS指针不是权重本体。

## 加载

先按仓库根目录说明下载固定版本 `f75b40e3da2003cdd6e1829b1f420ca70797c34e` 的 Qwen2.5-Omni-3B。与训练实现一致，将适配器加载到 **Thinker** 模型，而不是 Talker 或完整 Omni 顶层模块：

```python
import torch
from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from peft import PeftModel

base_path = "models/Qwen2.5-Omni-3B"
adapter_path = "models/EquiAffect-3B"
processor = AutoProcessor.from_pretrained(base_path)
base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    base_path,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
)
model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
model.eval()
```

参考原实验环境：Transformers 4.52.2、PEFT 0.15.2、PyTorch 2.5.1。多模态输入处理及提示模板见 `evaluation/` 与 `training/`。以上加载片段根据训练时使用的模型类和标准 PEFT 接口整理，本次归档未重新执行 GPU 加载或推理。

权重文件下载后可运行 `python scripts/verify_equiaffect.py` 检查。使用与分享需遵守基座 Qwen Research License。
