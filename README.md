# Tiny Text-to-SVG Model

A minimal, self-contained pipeline that treats SVG as a text-generation problem:
you type a caption like `cat icon`, the model generates SVG code, character by character.

## What's in here

| File | Purpose |
|---|---|
| `build_dataset.py` | Clones/parses icon libraries, normalizes SVGs, writes `data/dataset_raw.jsonl` |
| `train_tokenizer.py` | Trains a byte-level BPE tokenizer on the caption+SVG corpus |
| `model.py` | Tiny GPT-style decoder-only transformer (~1.2M params by default) |
| `train.py` | CPU-friendly training loop, resumable, checkpoints periodically |
| `generate.py` | Samples SVG from a text prompt, checks it's valid XML, saves it |
| `data/dataset_raw.jsonl` | 7,827 cleaned (caption, SVG) pairs, ready to train on |
| `tokenizer/` | The trained tokenizer (vocab.json + merges.txt) |
| `ckpt.pt` | A checkpoint at step 460 — **early, not a finished model** (see below) |

## Where the data came from

Cloned three MIT-licensed icon sets from GitHub and normalized every SVG (stripped
IDs/classes/comments, rounded coordinates, minified):
- **Feather** (287 icons) — caption from filename
- **Bootstrap Icons** (2,085 icons) — caption from filename
- **Tabler Icons** (6,146 icons, outline + filled) — caption from the `tags:` comment
  each file ships with, e.g. `tags: [baking, birthday, cake, ...]`

After dedup + length filtering (40–700 chars): **7,827 examples**, median SVG length
~387 characters.

## Honest status of `ckpt.pt`

I trained this checkpoint for 460 steps on a **single CPU core** (this sandbox's limit)
just to prove the whole pipeline works. Run `generate.py` on it and you'll see:

- ✅ It already nails the SVG *wrapper* syntax perfectly — `xmlns`, `viewBox`,
  `stroke-linecap`, etc. — because that's highly repetitive across the dataset.
- ❌ It does **not** yet produce valid path data (`d="..."`) — the `M`/`L`/`A`/`Z`
  path-command grammar and matching numeric structure needs a lot more training to
  become consistent. Right now generated paths are XML-invalid.

This is expected at this loss level (~1.5–1.8) — it's the very beginning of training,
not a failure of the approach. The path grammar is the hard part; the wrapper syntax
was the easy part.

## How to actually train this to something usable

You'll get real mileage on your own machine (more cores > this sandbox's 1 core):

```bash
pip install torch tokenizers lxml --break-system-packages

# keep training from where I left off
python3 train.py --steps 20000 --batch_size 32 --log_every 100 --save_every 200 --resume
```

- Watch `train_loss` / `val_loss` in the printout. Also periodically run:
  ```bash
  python3 generate.py --prompt "cat icon" --n_samples 8
  ```
  and check the `N/8 valid XML` count at the end. **That fraction climbing toward 8/8
  is your first real milestone** — it means the model has learned path syntax.
  After that, quality of the actual shapes is the next thing to watch.
- If training gets interrupted, `--resume` picks up from the last saved step
  (checkpoints save every `--save_every` steps, not just at the end).
- If you have more RAM/cores, bump `--batch_size` (32–64) before bumping model size —
  bigger batches are the cheapest way to speed up convergence here.

## Scaling up later

- **More/better data**: add more MIT icon sets (Lucide, Iconoir, Phosphor Icons) the
  same way `build_dataset.py` does — just add their GitHub URLs and a caption rule.
- **Bigger model**: `train.py --n_layer 6 --n_head 6 --n_embd 192` etc., once you've
  confirmed the small one is learning — bump size only after the loss curve looks healthy.
- **Beyond icons**: photorealistic "SVG photos" aren't realistic for this approach —
  real photos as SVG are thousands of path commands, far beyond what a small
  text model can learn. If you want to go past flat icons/illustrations, the research
  directions to look at are diffusion-based vector graphics (e.g. VectorFusion-style
  approaches) or larger transformer models trained on curated vector-art datasets
  (e.g. the StarVector / IconShop papers) — different, heavier approach than this repo.

## Licensing note

Feather, Bootstrap Icons, and Tabler Icons are all MIT-licensed — fine to train on,
attribution appreciated but not required.
