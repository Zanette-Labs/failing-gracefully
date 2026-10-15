import importlib.metadata as md
import os
import subprocess
import torch
import transformer_engine
import sglang
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH

for name in ["torch", "sglang", "megatron-core", "transformer-engine", "flash-attn",
             "flashinfer-python", "ray", "transformers", "wandb", "numpy", "torch_memory_saver"]:
    print(name, md.version(name), flush=True)
for path in ["/root/Megatron-LM", "/sgl-workspace/sglang"]:
    print(path, subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip())
print("HOME", os.path.expanduser("~"))
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA device count", torch.cuda.device_count())
assert torch.cuda.device_count() == 8
for i in range(8):
    x = torch.ones(8, device=f"cuda:{i}")
    assert (x + 1).sum().item() == 16
    print(i, torch.cuda.get_device_name(i), "tensor operation OK", flush=True)

assert subprocess.check_output(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], text=True).strip() == "8da71333cbdb131a16b52072f6a67295ec764071"
assert subprocess.check_output(["git", "-C", "/root/Megatron-LM", "rev-parse", "HEAD"], text=True).strip() == "a381c7da699e4c991f0906434ab2a2c8bd18c1e5"
