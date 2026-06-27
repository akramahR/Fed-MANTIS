import torch
from mantis.architecture import Mantis8M

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Mantis8M(device=device)  # or: .from_pretrained("paris-noah/Mantis-8M")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
bytes_per_param = next(model.parameters()).element_size()  # 4 for fp32, 2 for fp16

print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
print(f"Weight memory ≈ {total_params * bytes_per_param / 1e6:.2f} MB "
      f"(dtype={next(model.parameters()).dtype})")
