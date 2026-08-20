import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability: sm_%d%d" % torch.cuda.get_device_capability(0))
print("compiled arch list:", torch.cuda.get_arch_list())

x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().sum().item()
print("bf16 matmul on GPU ok, sum =", y)
print("free/total VRAM (GB):", [round(v / 1e9, 2) for v in torch.cuda.mem_get_info()])
