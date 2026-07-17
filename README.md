# Savage SVG Generator

Savage is a state-of-the-art text-to-SVG generation pipeline that fine-tunes **Qwen2.5-Coder-1.5B** to natively output highly detailed, multi-color vector illustrations and stencils based on text prompts.

Instead of generating pixels or relying on diffusion models, Savage treats scalable vector graphics (SVG) generation purely as a language modeling task, teaching an LLM the exact syntax and coordinate math needed to draw beautiful scalable vectors.

## How It Works

The magic of Savage lies in its custom data engine and efficient fine-tuning pipeline:

1. **Dataset Engine (`build_dataset.py` & `image_to_svg.py`)**
   We don't train on simple, pre-existing icons. Instead, we use `vtracer` to programmatically trace and vectorize massive photo datasets (like CIFAR-10 or Japanese Photos) into highly detailed, 3-tone posterized SVGs. The engine intelligently optimizes the SVGs to fit perfectly within the context window of modern LLMs (under 64,000 characters).
2. **LoRA Fine-Tuning (`train.py`)**
   We fine-tune Qwen2.5-Coder-1.5B using Parameter-Efficient Fine-Tuning (LoRA) on the generated dataset. The script supports local execution (MPS/CUDA) and is heavily optimized to run on **Kaggle** (via the `--kaggle` flag).
3. **High-Speed Generation (`generate.py`)**
   We use Apple's MLX (`mlx-lm`) to achieve incredible generation speeds on Apple Silicon, while seamlessly falling back to HuggingFace Transformers for other platforms.

## Core Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Automatically downloads HuggingFace image datasets, vectorizes them into highly detailed SVGs, and outputs `dataset_raw.jsonl`. |
| `image_to_svg.py` | The core vectorization engine under the hood. It removes backgrounds, posterizes images, and uses `vtracer` to generate beautiful, low-byte SVGs. |
| `train.py` | The LoRA fine-tuning script for Qwen2.5-Coder-1.5B. Supports multi-GPU, MPS, and Kaggle environments out of the box. |
| `generate.py` | The inference engine. Pass a prompt and it will stream the generated SVG to your console and save it to disk. Supports both `mlx` and `transformers`. |
| `fuse.py` | Utility to permanently fuse the trained LoRA weights into the base Qwen model for faster inference. |

## Quickstart

### 1. Build your dataset
```bash
python3 build_dataset.py
```
This will start tracing images into text-ready SVGs and append them to `dataset_raw.jsonl`.

### 2. Train the model
You can train locally or on Kaggle (T4x2 is recommended).
```bash
# Local training
python3 train.py

# Kaggle training
python3 train.py --kaggle
```

### 3. Generate SVGs
Once trained, use the generation script to draw anything:
```bash
python3 generate.py --prompt "a beautiful orange cat" --engine mlx
```
*(Use `--engine transformers` if you are not on a Mac!)*

## Future Roadmap
- Increase dataset size with high-quality captions (e.g., BLIP-generated captions).
- Scale up the base model (e.g., 1.5B or 7B coders).
- Improve the color-quantization logic for even richer SVGs while maintaining low token counts.
