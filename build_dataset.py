import json
import tqdm
from datasets import load_dataset
import numpy as np
import cv2

import image_to_svg

def trace_image(img_pil):
    try:

        svg_str = image_to_svg.convert(
            input_path=img_pil,
            output_path=None,
            target_kb=5.0,
            cutout=True,
            posterize=3,
            verbose=False,
        )
        return svg_str
    except Exception as e:
        print(f"Error tracing image: {e}")
        return ""

def build_dataset(
    dataset_name,
    out_path="dataset_raw.jsonl",
    target_count=5000,
    image_col="image",
    text_col="text",
    streaming=True,
):
    print(f"Downloading photo dataset {dataset_name} for {target_count} photos...")
    cifar10_classes = None
    try:
        ds = load_dataset(dataset_name, split="train", streaming=streaming)
    except Exception as e:
        print(f"Warning: failed to load dataset via `datasets`: {e}")
        # Fallback: handle CIFAR-10 via torchvision if HF Hub parsing fails
        if dataset_name.lower().startswith("cifar10"):
            try:
                from torchvision.datasets import CIFAR10

                cifar = CIFAR10(root="./.cache", train=True, download=True)
                cifar10_classes = cifar.classes

                def _cifar_gen():
                    for img, label in cifar:
                        yield {image_col: img, text_col: label}

                ds = _cifar_gen()
                streaming = False
            except Exception as e2:
                print(f"CIFAR fallback failed: {e2}")
                raise
        else:
            raise

    count = 0

    with open(out_path, "a") as f:
        pbar = tqdm.tqdm(total=target_count)
        for row in ds:
            try:
                img = row[image_col]
                caption = row[text_col]

                if isinstance(caption, int):
                    if cifar10_classes is not None:
                        caption = cifar10_classes[caption]
                    else:
                        try:
                            caption = ds.features[text_col].int2str(caption)
                        except Exception:
                            caption = str(caption)
                else:
                    caption = str(caption)

                svg_str = trace_image(img)

                if len(svg_str) < 64000:
                    f.write(
                        json.dumps(
                            {"caption": caption, "svg": svg_str, "source": dataset_name}
                        )
                        + "\n"
                    )
                    count += 1
                    pbar.update(1)

                if count >= target_count:
                    break
            except Exception as e:

                continue

        pbar.close()
    print(
        f"Successfully vectorized {count} real photos from {dataset_name} to {out_path}!"
    )

if __name__ == "__main__":

    build_dataset(
        "cifar10",
        target_count=20000,
        image_col="img",
        text_col="label",
        streaming=False,
    )

    build_dataset(
        "nbeerbower/japanese-photos-captioned",
        target_count=5000,
        image_col="image",
        text_col="caption",
        streaming=True,
    )
