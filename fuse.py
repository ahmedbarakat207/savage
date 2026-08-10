import os
import argparse
import sys
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")

def main():
    parser = argparse.ArgumentParser(description="Fuse LoRA adapter weights into base model.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-3B", help="Base model HuggingFace ID or local path")
    parser.add_argument("--adapter_path", type=str, default="./savage-1", help="Path to trained LoRA adapter")
    parser.add_argument("--output_dir", type=str, default="./savage-fused", help="Destination path for fused model")
    parser.add_argument("--kaggle", action="store_true", help="Use Kaggle default paths")
    args = parser.parse_args()

    if args.kaggle:
        if args.adapter_path == "./savage-1":
            args.adapter_path = "/kaggle/working/savage-1"
        if args.output_dir == "./savage-fused":
            args.output_dir = "/kaggle/working/savage-fused"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer from '{args.adapter_path}' (fallback: '{args.base_model}')...")
    tokenizer_source = args.adapter_path if os.path.exists(args.adapter_path) else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    print(f"Loading base model from '{args.base_model}'...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        trust_remote_code=True
    )

    # Resize base model embeddings to match custom tokens added during training
    base.resize_token_embeddings(len(tokenizer))

    print(f"Loading LoRA adapter from '{args.adapter_path}'...")
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, args.adapter_path)
    except ImportError as e:
        if "torchao" in str(e):
            print("\n" + "="*70)
            print("ERROR: Incompatible torchao version detected by PEFT.")
            print("To fix this in Colab / Linux, run:")
            print("    pip install --upgrade torchao")
            print("or")
            print("    pip uninstall -y torchao")
            print("="*70 + "\n")
        raise e

    print("Fusing weights (merge_and_unload)...")
    model = model.merge_and_unload()

    print(f"Saving fused model to '{args.output_dir}'...")
    model.save_pretrained(args.output_dir)

    print(f"Saving tokenizer to '{args.output_dir}'...")
    tokenizer.save_pretrained(args.output_dir)
    print("Done! Fused model is ready for inference.")

if __name__ == "__main__":
    main()

