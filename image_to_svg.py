import argparse
import io
import os
import re
import sys
import tempfile

import numpy as np
from PIL import Image, ImageFilter

try:
    import vtracer
except ImportError:
    sys.exit("Missing dependency: pip install vtracer --break-system-packages")

try:
    import cairosvg
    from skimage.metrics import structural_similarity as ssim

    HAVE_QUALITY_TOOLS = True
except (ImportError, OSError) as e:
    print(f"Warning: Quality tools disabled due to load error: {e}")
    HAVE_QUALITY_TOOLS = False

try:
    from skimage.segmentation import flood
    from scipy.cluster.vq import kmeans, vq
except ImportError:
    pass

PRESETS = [
    dict(
        max_dim=512,
        blur=0.0,
        color_precision=8,
        layer_difference=8,
        filter_speckle=2,
        corner_threshold=40,
        length_threshold=3.0,
        splice_threshold=30,
        path_precision=2,
        mode="spline",
    ),
    dict(
        max_dim=384,
        blur=0.3,
        color_precision=7,
        layer_difference=10,
        filter_speckle=3,
        corner_threshold=45,
        length_threshold=3.5,
        splice_threshold=35,
        path_precision=2,
        mode="spline",
    ),
    dict(
        max_dim=320,
        blur=0.5,
        color_precision=6,
        layer_difference=12,
        filter_speckle=4,
        corner_threshold=50,
        length_threshold=4.0,
        splice_threshold=40,
        path_precision=1,
        mode="spline",
    ),
    dict(
        max_dim=256,
        blur=0.6,
        color_precision=6,
        layer_difference=14,
        filter_speckle=6,
        corner_threshold=55,
        length_threshold=4.5,
        splice_threshold=45,
        path_precision=1,
        mode="spline",
    ),
    dict(
        max_dim=200,
        blur=0.6,
        color_precision=7,
        layer_difference=24,
        filter_speckle=18,
        corner_threshold=70,
        length_threshold=6.0,
        splice_threshold=50,
        path_precision=1,
        mode="spline",
    ),
    dict(
        max_dim=170,
        blur=0.7,
        color_precision=7,
        layer_difference=36,
        filter_speckle=26,
        corner_threshold=78,
        length_threshold=7.0,
        splice_threshold=55,
        path_precision=1,
        mode="spline",
    ),
    dict(
        max_dim=160,
        blur=0.8,
        color_precision=7,
        layer_difference=40,
        filter_speckle=28,
        corner_threshold=80,
        length_threshold=7.5,
        splice_threshold=58,
        path_precision=1,
        mode="spline",
    ),
    dict(
        max_dim=140,
        blur=0.8,
        color_precision=7,
        layer_difference=40,
        filter_speckle=30,
        corner_threshold=82,
        length_threshold=8.0,
        splice_threshold=60,
        path_precision=1,
        mode="spline",
    ),
    dict(
        max_dim=130,
        blur=0.8,
        color_precision=7,
        layer_difference=42,
        filter_speckle=29,
        corner_threshold=80,
        length_threshold=7.3,
        splice_threshold=58,
        path_precision=1,
        mode="spline",
    ),
    dict(
        max_dim=120,
        blur=0.9,
        color_precision=7,
        layer_difference=48,
        filter_speckle=34,
        corner_threshold=88,
        length_threshold=9.0,
        splice_threshold=65,
        path_precision=1,
        mode="spline",
    ),
    dict(
        max_dim=110,
        blur=1.0,
        color_precision=6,
        layer_difference=44,
        filter_speckle=32,
        corner_threshold=85,
        length_threshold=8.5,
        splice_threshold=62,
        path_precision=1,
        mode="spline",
    ),
    dict(
        max_dim=100,
        blur=1.1,
        color_precision=5,
        layer_difference=36,
        filter_speckle=28,
        corner_threshold=85,
        length_threshold=8.0,
        splice_threshold=65,
        path_precision=0,
        mode="polygon",
    ),
    dict(
        max_dim=90,
        blur=1.3,
        color_precision=4,
        layer_difference=32,
        filter_speckle=24,
        corner_threshold=85,
        length_threshold=8.0,
        splice_threshold=70,
        path_precision=0,
        mode="polygon",
    ),
]

NUM_RE = re.compile(r"-?\d+\.\d+")

CUTOUT_PRESETS = [
    dict(
        max_dim=182,
        filter_speckle=4,
        corner_threshold=35,
        length_threshold=2.5,
        splice_threshold=25,
        path_precision=1,
        mode="polygon",
    ),
    dict(
        max_dim=170,
        filter_speckle=6,
        corner_threshold=40,
        length_threshold=3.0,
        splice_threshold=30,
        path_precision=1,
        mode="polygon",
    ),
    dict(
        max_dim=150,
        filter_speckle=8,
        corner_threshold=45,
        length_threshold=3.5,
        splice_threshold=35,
        path_precision=1,
        mode="polygon",
    ),
    dict(
        max_dim=130,
        filter_speckle=10,
        corner_threshold=50,
        length_threshold=4.0,
        splice_threshold=40,
        path_precision=1,
        mode="polygon",
    ),
    dict(
        max_dim=120,
        filter_speckle=12,
        corner_threshold=55,
        length_threshold=4.5,
        splice_threshold=42,
        path_precision=1,
        mode="polygon",
    ),
    dict(
        max_dim=112,
        filter_speckle=18,
        corner_threshold=64,
        length_threshold=5.8,
        splice_threshold=50,
        path_precision=1,
        mode="polygon",
    ),
    dict(
        max_dim=110,
        filter_speckle=14,
        corner_threshold=58,
        length_threshold=5.0,
        splice_threshold=45,
        path_precision=1,
        mode="polygon",
    ),
    dict(
        max_dim=100,
        filter_speckle=16,
        corner_threshold=60,
        length_threshold=5.5,
        splice_threshold=48,
        path_precision=1,
        mode="polygon",
    ),
    dict(
        max_dim=95,
        filter_speckle=18,
        corner_threshold=63,
        length_threshold=6.0,
        splice_threshold=50,
        path_precision=0,
        mode="polygon",
    ),
    dict(
        max_dim=85,
        filter_speckle=20,
        corner_threshold=65,
        length_threshold=6.5,
        splice_threshold=55,
        path_precision=0,
        mode="polygon",
    ),
]

def remove_background(img, tolerance=18):
    rgb = np.array(img.convert("RGB"))
    gray = rgb.astype(np.int16).mean(axis=2)
    h, w = gray.shape
    mask = np.zeros(gray.shape, dtype=bool)
    for seed in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        mask |= flood(gray, seed, tolerance=tolerance)
    alpha = (~mask).astype(np.uint8) * 255
    out = np.dstack([rgb, alpha])
    return Image.fromarray(out, "RGBA")

def to_grayscale(img):
    arr = np.array(img.convert("RGBA"))
    rgbf = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3]
    gray = (
        0.299 * rgbf[:, :, 0] + 0.587 * rgbf[:, :, 1] + 0.114 * rgbf[:, :, 2]
    ).astype(np.uint8)
    out = np.dstack([gray, gray, gray, alpha])
    return Image.fromarray(out, "RGBA")

def posterize_gray(img, n_levels, method="kmeans"):
    arr = np.array(img.convert("RGBA"))
    rgbf = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3]
    gray = 0.299 * rgbf[:, :, 0] + 0.587 * rgbf[:, :, 1] + 0.114 * rgbf[:, :, 2]

    if method == "kmeans" and HAVE_QUALITY_TOOLS:
        fg = gray[alpha > 0].astype(np.float64)
        if len(np.unique(fg)) > n_levels:
            centroids, _ = kmeans(fg, n_levels, seed=0)
            centroids = np.sort(centroids)
            idx, _ = vq(gray.flatten().astype(np.float64), centroids)
            post = centroids[idx].reshape(gray.shape).astype(np.uint8)
            post = np.where(alpha > 0, post, 0).astype(np.uint8)
        else:
            post = gray.astype(np.uint8)
    else:
        post = (np.round(gray / 255 * (n_levels - 1)) / (n_levels - 1) * 255).astype(
            np.uint8
        )

    out = np.dstack([post, post, post, alpha])
    return Image.fromarray(out, "RGBA")

def _round_numbers(text, ndigits):
    def _r(m):
        v = round(float(m.group(0)), ndigits)
        return str(int(v)) if v == int(v) else str(v)

    return NUM_RE.sub(_r, text)

def minify_svg(text, path_precision=1):
    text = re.sub(r">\s+<", "><", text.strip())
    text = _round_numbers(text, path_precision)
    text = re.sub(r"[ \t]+", " ", text)
    return text

def trace_with_preset(orig_img, preset, posterize_levels=None):
    w, h = orig_img.size
    scale = preset["max_dim"] / max(w, h)
    img = orig_img
    if scale < 1:
        img = img.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS
        )
    if preset.get("blur", 0) > 0:
        img = img.filter(ImageFilter.GaussianBlur(preset["blur"]))
    if posterize_levels:
        img = posterize_gray(img, posterize_levels)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        img.convert("RGBA").save(tmp_in.name)
        in_path = tmp_in.name
    out_path = in_path + ".svg"
    try:
        vtracer.convert_image_to_svg_py(
            in_path,
            out_path,
            "color",
            "stacked",
            preset["mode"],
            preset["filter_speckle"],
            preset["color_precision"],
            preset.get("layer_difference", 16),
            preset["corner_threshold"],
            preset["length_threshold"],
            10,
            preset["splice_threshold"],
            preset["path_precision"],
        )
        svg_text = open(out_path).read()
    finally:
        os.path.exists(in_path) and os.unlink(in_path)
        os.path.exists(out_path) and os.unlink(out_path)

    svg_text = minify_svg(svg_text, preset["path_precision"])
    return svg_text, img

def score_quality(orig_img, svg_text):
    if not HAVE_QUALITY_TOOLS:
        return None
    w, h = orig_img.size
    png_bytes = cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"), output_width=w, output_height=h
    )
    rendered = Image.open(io.BytesIO(png_bytes))
    if orig_img.mode == "RGBA":
        rendered_arr = np.array(rendered.convert("RGBA"))
        orig_arr = np.array(orig_img)
    else:
        rendered_arr = np.array(rendered.convert("RGB"))
        orig_arr = np.array(orig_img.convert("RGB"))
    return float(ssim(orig_arr, rendered_arr, channel_axis=2))

def convert(
    input_path,
    output_path=None,
    target_kb=5.0,
    verbose=True,
    preview_path=None,
    min_candidates=2,
    cutout=False,
    posterize=None,
    bg_tolerance=18,
):
    target_bytes = int(target_kb * 1024)
    if isinstance(input_path, str):
        orig_img = Image.open(input_path)
    else:
        orig_img = input_path
    if orig_img.mode in ("P", "RGBA"):
        orig_img = orig_img.convert("RGBA")
    else:
        orig_img = orig_img.convert("RGB")

    base_img = orig_img
    if cutout:
        base_img = remove_background(base_img, tolerance=bg_tolerance)
        if verbose:
            print("  applied background removal (cutout)")
    reference_img = to_grayscale(base_img) if posterize else base_img
    if posterize and verbose:
        print(
            f"  will posterize to {posterize} gray levels via k-means (per-preset, after resize)"
        )

    candidates = []
    active_presets = CUTOUT_PRESETS if posterize else PRESETS
    for i, preset in enumerate(active_presets):
        p = dict(preset)
        if posterize:

            p["color_precision"] = 8
            p["layer_difference"] = 4
            p.setdefault("blur", 0.0)
        svg_text, _ = trace_with_preset(base_img, p, posterize_levels=posterize)
        size = len(svg_text.encode("utf-8"))
        fits = size <= target_bytes
        if verbose:
            tag = "OK, fits" if fits else "too big"
            print(
                f"  preset {i:>2} | max_dim={p['max_dim']:>3} "
                f"colors=2^{p['color_precision']} mode={p['mode']:<7} "
                f"-> {size:>6} bytes  [{tag}]"
            )
        if fits:
            score = score_quality(reference_img, svg_text)
            candidates.append((score if score is not None else 0.0, size, svg_text, i))

            if len(candidates) >= min_candidates and size < target_bytes * 0.5:
                break

    if not candidates:

        fallback = dict(active_presets[-1])
        if posterize:
            fallback["color_precision"] = 8
            fallback["layer_difference"] = 4
            fallback.setdefault("blur", 0.0)
        svg_text, _ = trace_with_preset(base_img, fallback, posterize_levels=posterize)
        size = len(svg_text.encode("utf-8"))
        if verbose:
            print(
                f"WARNING: could not reach {target_kb}KB even at lowest quality "
                f"({size/1024:.1f}KB). This image may be too detailed/high-contrast "
                f"for this size budget -- writing best-effort result anyway."
            )
        if output_path:
            with open(output_path, "w") as f:
                f.write(svg_text)
        return svg_text

    best_score, best_size, best_svg, best_idx = max(candidates, key=lambda c: c[0])
    if output_path:
        with open(output_path, "w") as f:
            f.write(best_svg)

    if verbose:
        msg = f"Chosen preset {best_idx}: {best_size} bytes ({best_size/1024:.2f} KB)"
        if HAVE_QUALITY_TOOLS:
            msg += f", SSIM={best_score:.3f}"
        print(msg)

    if preview_path:
        w, h = reference_img.size
        png_bytes = cairosvg.svg2png(
            bytestring=best_svg.encode("utf-8"), output_width=w, output_height=h
        )
        with open(preview_path, "wb") as f:
            f.write(png_bytes)
        if verbose:
            print(f"Preview saved to {preview_path}")

    return best_svg

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, help="input image (jpg/png/bmp/etc)")
    ap.add_argument("--output", required=True, help="output .svg path")
    ap.add_argument(
        "--target-kb", type=float, default=5.0, help="max output size in KB (default 5)"
    )
    ap.add_argument(
        "--preview",
        action="store_true",
        help="also save a rendered PNG preview next to the output",
    )
    ap.add_argument(
        "--cutout",
        action="store_true",
        help="remove a roughly-uniform background (e.g. plain studio backdrop) to transparency",
    )
    ap.add_argument(
        "--bg-tolerance",
        type=int,
        default=18,
        help="color-distance tolerance for --cutout background detection (default 18)",
    )
    ap.add_argument(
        "--posterize",
        type=int,
        default=None,
        metavar="N",
        help="convert to grayscale posterized to N tone levels before tracing -- "
        "frees up most of the byte budget for shape detail instead of color "
        "(try 4-6; combine with --cutout for a clean 'sticker' look)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not HAVE_QUALITY_TOOLS:
        print(
            "Note: cairosvg/scikit-image not found, skipping SSIM-based quality "
            "scoring (falls back to first preset that fits). "
            "pip install cairosvg scikit-image for better results."
        )

    preview_path = None
    if args.preview:
        base, _ = os.path.splitext(args.output)
        preview_path = base + "_preview.png"

    convert(
        args.input,
        args.output,
        target_kb=args.target_kb,
        cutout=args.cutout,
        posterize=args.posterize,
        bg_tolerance=args.bg_tolerance,
        verbose=not args.quiet,
        preview_path=preview_path,
    )

if __name__ == "__main__":
    main()
