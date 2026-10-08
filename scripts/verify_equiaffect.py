"""Verify the trained adapter without loading the base model or using a GPU."""
import hashlib
import json
import math
import struct
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = root / "models" / "EquiAffect-3B"
manifest = json.loads((folder / "source_manifest.json").read_text(encoding="utf8"))
for name, expected in manifest["files"].items():
    data = (folder / name).read_bytes()
    assert len(data) == expected["bytes"], f"Size mismatch: {name}"
    assert hashlib.sha256(data).hexdigest() == expected["sha256"], f"Hash mismatch: {name}"

data = (folder / "adapter_model.safetensors").read_bytes()
header_length = struct.unpack("<Q", data[:8])[0]
header = json.loads(data[8:8 + header_length])
tensors = {k: v for k, v in header.items() if k != "__metadata__"}
config = json.loads((folder / "adapter_config.json").read_text(encoding="utf8"))
base_map = json.loads((root / "models/base_model_metadata/model.safetensors.index.json").read_text(encoding="utf8"))["weight_map"]
offsets = []
for key, tensor in tensors.items():
    assert tensor["dtype"] == "F32", key
    start, end = tensor["data_offsets"]
    assert end - start == math.prod(tensor["shape"]) * 4, key
    offsets.append((start, end))
    assert key.startswith("base_model.model.") and ".lora_" in key, key
    module = key.removeprefix("base_model.model.").split(".lora_")[0]
    assert "thinker." + module + ".weight" in base_map, key
    if key.endswith(".lora_A.weight"):
        partner = key.replace(".lora_A.weight", ".lora_B.weight")
        assert partner in tensors, key
        assert tensor["shape"][0] == config["r"], key
        assert tensors[partner]["shape"][1] == config["r"], key
offsets.sort()
assert offsets[0][0] == 0
assert all(a[1] == b[0] for a, b in zip(offsets, offsets[1:]))
assert offsets[-1][1] == len(data) - 8 - header_length
expected_modules = {k.removeprefix("thinker.").removesuffix(".weight") for k in base_map if k.startswith("thinker.") and k.endswith(".weight") and k.split(".")[-2] in config["target_modules"]}
actual_modules = {k.removeprefix("base_model.model.").split(".lora_")[0] for k in tensors}
assert expected_modules == actual_modules, "Adapter target coverage mismatch"
print(json.dumps({"model": "EquiAffect-3B", "files_verified": len(manifest["files"]), "tensors": len(tensors), "target_modules": len(actual_modules), "base_module_coverage": "complete", "sha256_match": True}, indent=2))
