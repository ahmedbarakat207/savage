import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B", 
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, "./savage-1")

print("Fusing weights...")
model = model.merge_and_unload()

print("Saving fused model to ./savage-fused...")
model.save_pretrained("./savage-fused")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", trust_remote_code=True)
tokenizer.save_pretrained("./savage-fused")
print("Done!")
