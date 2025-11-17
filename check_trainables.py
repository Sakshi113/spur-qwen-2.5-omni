# save as check_trainables.py and run: python check_trainables.py
from transformers import Qwen2_5OmniForConditionalGeneration
import torch

m = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B", trust_remote_code=True, torch_dtype=torch.float16
)

# Simulate your YAML freeze plan:
# 1) freeze everything
for p in m.parameters():
    p.requires_grad = False
# 2) unfreeze spatial/audio encoder + projector
for n, p in m.named_parameters():
    if n.startswith("thinker.audio_tower") or n.startswith("thinker.visual.merger"):
        p.requires_grad = True

audio_trainable = audio_total = 0
proj_trainable = proj_total = 0

for n, p in m.named_parameters():
    if n.startswith("thinker.audio_tower"):
        audio_total += p.numel()
        if p.requires_grad:
            audio_trainable += p.numel()
    if n.startswith("thinker.visual.merger"):
        proj_total += p.numel()
        if p.requires_grad:
            proj_trainable += p.numel()

print("audio_tower trainable params:", audio_trainable, "/", audio_total)
print("visual.merger trainable params:", proj_trainable, "/", proj_total)
print("examples of trainable in audio tower:",
      [n for n,p in list(m.named_parameters()) if n.startswith("thinker.audio_tower") and p.requires_grad][:5])
