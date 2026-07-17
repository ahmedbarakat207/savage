import os

os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import argparse
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import LoraConfig, get_peft_model

def load_data(file_path):
    data = {"text": []}
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                caption = item.get("caption", "").strip()
                svg = item.get("svg", "").strip()
                if caption and svg:
                    prompt = f"<|im_start|>user\nGenerate an SVG for: {caption}<|im_end|>\n<|im_start|>assistant\n{svg}<|im_end|>"
                    data["text"].append(prompt)
            except:
                pass
    return Dataset.from_dict(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--kaggle", action="store_true")
    parser.add_argument("--colab-fast", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    model_id = "Qwen/Qwen2.5-Coder-0.5B"

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto" if args.kaggle else (device if device != "mps" else None),
        trust_remote_code=True,
    )
    if not args.kaggle and device == "mps":
        model.to("mps")

    if args.download_only:
        print(f"Successfully downloaded the model '{model_id}' to the Hugging Face cache.")
        print("Exiting because --download-only was specified.")
        exit(0)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if args.kaggle:
        dataset_path = "/kaggle/input/savage-svg-dataset/dataset_raw.jsonl"
    else:
        dataset_path = "dataset_raw.jsonl"

    dataset = load_data(dataset_path).train_test_split(test_size=0.05)

    if args.colab_fast:
        max_len = 2048
    else:
        max_len = 4096 if args.kaggle else 16384

    tokenized_datasets = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=max_len),
        batched=True,
        remove_columns=["text"],
    )

    if args.quick_test:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(
            range(min(100, len(tokenized_datasets["train"])))
        )
        tokenized_datasets["test"] = tokenized_datasets["test"].select(
            range(min(10, len(tokenized_datasets["test"])))
        )
    elif args.colab_fast:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(
            range(min(2000, len(tokenized_datasets["train"])))
        )

    out_dir = "/kaggle/working/savage-lora" if args.kaggle else "./savage-lora"
    final_dir = "/kaggle/working/savage-1" if args.kaggle else "./savage-1"

    if args.colab_fast:
        batch_size = 2
        grad_accum = 8
        max_steps = 200
        logging_steps = 5
    elif args.kaggle:
        batch_size = 2
        grad_accum = 4
        max_steps = 500
        logging_steps = 10
    else:
        batch_size = 1
        grad_accum = 8
        max_steps = 500
        logging_steps = 1

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        logging_steps=logging_steps,
        max_steps=10 if args.quick_test else max_steps,
        save_steps=5 if args.quick_test else 100,
        eval_strategy="steps",
        eval_steps=5 if args.quick_test else 100,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    resume_ckpt = None
    if args.resume and os.path.isdir(out_dir):
        if any(d.startswith("checkpoint") for d in os.listdir(out_dir)):
            resume_ckpt = True
            print(f"Found existing checkpoints in {out_dir}, resuming training...")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

if __name__ == "__main__":
    main()
