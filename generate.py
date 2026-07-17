import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
import argparse
import time

def generate_transformers(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
    from peft import PeftModel

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    if (args.base_model.startswith("./") or args.base_model.startswith("/")) and not os.path.exists(args.base_model):
        print(f"Error: The local model path '{args.base_model}' does not exist.")
        print("Did you forget to run 'python fuse.py' to create the fused model?")
        print("Alternatively, pass the base model explicitly: --base_model Qwen/Qwen2.5-Coder-0.5B")
        exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, trust_remote_code=True
    )

    if "fused" in args.base_model.lower():
        model = base
    else:
        try:
            model = PeftModel.from_pretrained(base, args.lora_path)
        except:
            model = base

    model.to(device)
    model.eval()

    # Prime the model to skip the VTracer comment entirely and start drawing the SVG
    primer = '<?xml version="1" encoding="UTF-8"?><!-- Generator: visioncortex VTracer 0.6.12 -->\n<svg'
    prompt = f"<|im_start|>user\nGenerate an SVG for: {args.prompt}<|im_end|>\n<|im_start|>assistant\n{primer}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if not args.no_stream else None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    return generated_text

def generate_mlx(args):
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("mlx-lm not found! Install it with: pip install mlx-lm")
        exit(1)

    try:
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(args.temperature)
    except ImportError:
        sampler = None

    if (args.base_model.startswith("./") or args.base_model.startswith("/")) and not os.path.exists(args.base_model):
        print(f"Error: The local model path '{args.base_model}' does not exist.")
        print("Did you forget to run 'python fuse.py' to create the fused model?")
        print("Alternatively, pass the base model explicitly: --base_model Qwen/Qwen2.5-Coder-0.5B")
        exit(1)

    model, tokenizer = load(args.base_model)
    prompt = f"<|im_start|>user\nGenerate an SVG for: {args.prompt}<|im_end|>\n<|im_start|>assistant\n"

    start_time = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_new_tokens,
        sampler=sampler,
        verbose=True,
    )

    num_tokens = len(tokenizer.encode(response))
    print(f"\nPerformance: {num_tokens / (time.time() - start_time):.2f} tokens/second")
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default="./savage-1")
    parser.add_argument("--base_model", type=str, default="./savage-fused")
    parser.add_argument(
        "--engine", type=str, choices=["mlx", "transformers"], default="mlx"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--no-stream", action="store_true", help="Disable real-time terminal output to prevent freezing")
    args = parser.parse_args()

    if args.engine == "mlx":
        response = generate_mlx(args)
    else:
        response = generate_transformers(args)

    out_file = args.prompt.replace(" ", "_") + ".svg"
    
    svg_content = response.replace("```xml", "").replace("```svg", "").replace("```", "").strip()
    
    # We must prepend the '<svg' that we primed the model with, ensuring a space
    if not svg_content.startswith("<svg"):
        svg_content = "<svg " + svg_content.lstrip()

    if "<svg" in svg_content:
        svg_content = svg_content[svg_content.find("<svg"):]
    if "</svg>" in svg_content:
        svg_content = svg_content[:svg_content.find("</svg>") + 6]
        
    with open(out_file, "w") as f:
        f.write(svg_content)

if __name__ == "__main__":
    main()
