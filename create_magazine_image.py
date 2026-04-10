#!/usr/bin/env python
"""Create a magazine-style figure with one sample image per dataset.

Architecture
------------
  DATASET_MODALITY  dict: dataset name  → modality string
  MODALITY_RENDERERS dict: modality string → render function
  standardize()           PIL.Image      → fixed-size RGB PNG
  load_and_render()       dataset name   → PIL.Image (or None on failure)
  generate_latex()        rendered dict  → LaTeX figure string
  main()                  CLI entry-point

Outputs (written to --output-dir, default "dataset_samples/"):
  {DatasetName}.png      one standardised PNG per dataset
  magazine_figure.tex    paste-ready LaTeX figure for Overleaf

Usage
-----
  python create_magazine_image.py
  python create_magazine_image.py --output-dir figs/samples --ncols 7 --sample-idx 42
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Short display labels for LaTeX (keep ≤ 12 chars so they fit under the image)
# ---------------------------------------------------------------------------
DATASET_LABELS: dict[str, str] = {
    "ArabicCharacters": "Ar.~Chars.",
    "ArabicDigits":     "Ar.~Digits",
    "AWA2":             "AwA2",
    "Beans":            "Beans",
    "Cars196":          "Cars196",
    "Cars3D":           "Cars3D",
    "CIFAR10":          "CIFAR-10",
    "CIFAR10C":         "CIFAR-10-C",
    "CIFAR100":         "CIFAR-100",
    "CIFAR100C":        "CIFAR-100-C",
    "CLEVRER":          "CLEVRER",
    "Country211":       "Country211",
    "CUB200":           "CUB-200",
    "DSprites":         "dSprites",
    "DSpritesColor":    "dSpr.-Color",
    "DSpritesNoise":    "dSpr.-Noise",
    "DSpritesScream":   "dSpr.-Scream",
    "DTD":              "DTD",
    "EMNIST":           "EMNIST",
    "FacePointing":     "FacePointing",
    "FashionMNIST":     "F-MNIST",
    "FGVCAircraft":     "Aircraft",
    "Flowers102":       "Flowers102",
    "Food101":          "Food101",
    "Galaxy10Decal":    "Galaxy10",
    "HASYv2":           "HASYv2",
    "KMNIST":           "K-MNIST",
    "Linnaeus5":        "Linnaeus5",
    "MedMNIST":         "MedMNIST",
    "NotMNIST":         "notMNIST",
    "RockPaperScissor": "Rock-Paper-Sc.",
    "Shapes3D":         "Shapes3D",
    "SmallNORB":        "SmallNORB",
    "STL10":            "STL-10",
    "SVHN":             "SVHN",
    "ImageNet1K":       "ImageNet-1K",
    "ImageNet100":      "ImageNet-100",
    "ImageNet10":       "ImageNet-10",
    "TinyImagenet":     "TinyImgNet",
    "TinyImagenetC":    "TinyImgNet-C",
}

# ---------------------------------------------------------------------------
# Ordered dataset list – grouped by modality for a coherent visual layout
# ---------------------------------------------------------------------------
DATASET_ORDER: list[str] = [
    # Standard RGB benchmarks
    "CIFAR10", "CIFAR100", "CIFAR10C", "CIFAR100C",
    "STL10", "SVHN", "TinyImagenet", "TinyImagenetC",
    "ImageNet1K", "ImageNet100", "ImageNet10",
    # Natural images: scenes / plants / food / textures
    "Food101", "Flowers102", "Beans", "DTD",
    "Linnaeus5", "Country211", "Galaxy10Decal", "RockPaperScissor",
    # Fine-grained recognition
    "CUB200", "Cars196", "FGVCAircraft", "AWA2",
    # Synthetic / disentangled representations
    "DSprites", "DSpritesColor", "DSpritesNoise", "DSpritesScream",
    "Cars3D", "Shapes3D",
    # Grayscale / character images
    "FashionMNIST", "KMNIST", "EMNIST", "NotMNIST",
    "ArabicCharacters", "ArabicDigits", "HASYv2", "SmallNORB",
    # Specialised
    "FacePointing", "MedMNIST", "CLEVRER",
]

# ---------------------------------------------------------------------------
# Dataset → modality string
# ---------------------------------------------------------------------------
DATASET_MODALITY: dict[str, str] = {
    # Natural colour images
    "AWA2":             "image_rgb",
    "Beans":            "image_rgb",
    "Cars196":          "image_rgb",
    "Cars3D":           "image_rgb",
    "CIFAR10":          "image_rgb",
    "CIFAR10C":         "image_rgb",
    "CIFAR100":         "image_rgb",
    "CIFAR100C":        "image_rgb",
    "Country211":       "image_rgb",
    "CUB200":           "image_rgb",
    "DTD":              "image_rgb",
    "DSpritesColor":    "image_rgb",
    "DSpritesNoise":    "image_rgb",
    "DSpritesScream":   "image_rgb",
    "FacePointing":     "image_rgb",
    "FGVCAircraft":     "image_rgb",
    "Flowers102":       "image_rgb",
    "Food101":          "image_rgb",
    "Galaxy10Decal":    "image_rgb",
    "ImageNet1K":       "image_rgb",
    "ImageNet100":      "image_rgb",
    "ImageNet10":       "image_rgb",
    "Linnaeus5":        "image_rgb",
    "RockPaperScissor": "image_rgb",
    "Shapes3D":         "image_rgb",
    "STL10":            "image_rgb",
    "SVHN":             "image_rgb",
    "TinyImagenet":     "image_rgb",
    "TinyImagenetC":    "image_rgb",
    # Grayscale images
    "ArabicCharacters": "image_gray",
    "ArabicDigits":     "image_gray",
    "EMNIST":           "image_gray",
    "FashionMNIST":     "image_gray",
    "HASYv2":           "image_gray",
    "KMNIST":           "image_gray",
    "NotMNIST":         "image_gray",
    # SmallNORB: stereo pair → uses left_image / right_image keys
    "SmallNORB":        "image_stereo",
    # Binary synthetic (dSprites white shapes on black)
    "DSprites":         "image_binary",
    # Medical images (may be volumetric)
    "MedMNIST":         "image_medical",
    # Video
    "CLEVRER":          "video",
}

# Extra constructor kwargs for datasets that require a config name
DATASET_KWARGS: dict[str, dict] = {
    "EMNIST":     {"config_name": "balanced"},
    "ImageNet1K": {"streaming": True},
    "ImageNet100":{"streaming": True},
    "ImageNet10": {"streaming": True},
    "MedMNIST":   {"config_name": "pathmnist"},  # 2-D, colourful path-histology
}

# HuggingFace Hub fallbacks for datasets that are too large or fail to download.
# Maps dataset name -> (hub_id, image_key) for streaming a single sample.
HF_STREAMING_FALLBACKS: dict[str, tuple[str, str]] = {
    "ImageNet1K":   ("ILSVRC/imagenet-1k", "image"),
    "ImageNet100":  ("clane9/imagenet-100", "image"),
    "KMNIST":       ("tanganke/kmnist", "image"),
    "Shapes3D":     ("eurecom-ds/shapes3d", "image"),
    "TinyImagenet": ("Maysee/tiny-imagenet", "image"),
}

# Datasets that should prefer HF streaming over local download (to avoid huge downloads)
PREFER_STREAMING: set[str] = {"ImageNet1K", "ImageNet100"}

# ---------------------------------------------------------------------------
# Modality render functions  (sample dict → PIL.Image)
# ---------------------------------------------------------------------------

def _to_pil(img) -> Image.Image:
    """Convert a HuggingFace image field (PIL or numpy array) to PIL."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L")
        return Image.fromarray(arr)
    raise TypeError(f"Cannot convert {type(img)} to PIL.Image")


def render_image_rgb(sample: dict) -> Image.Image:
    return _to_pil(sample["image"]).convert("RGB")


def render_image_gray(sample: dict) -> Image.Image:
    # Convert to grayscale then back to RGB so all output images are uniform
    return _to_pil(sample["image"]).convert("L").convert("RGB")


def render_image_stereo(sample: dict) -> Image.Image:
    """SmallNORB-style: pick the left camera image."""
    key = "left_image" if "left_image" in sample else "image"
    return _to_pil(sample[key]).convert("L").convert("RGB")


def render_image_binary(sample: dict) -> Image.Image:
    """Binary images (e.g. dSprites {0,1}): scale to full 0–255 range."""
    raw = sample["image"]
    if isinstance(raw, np.ndarray) and raw.max() <= 1:
        raw = (raw * 255).astype(np.uint8)
    return _to_pil(raw).convert("RGB")


def render_image_medical(sample: dict) -> Image.Image:
    """Medical images: handle 3-D volumes (D, H, W) by taking the middle slice."""
    raw = sample["image"]
    if isinstance(raw, np.ndarray) and raw.ndim == 3 and raw.shape[-1] > 4:
        # Volumetric array (D, H, W) – take axial middle slice
        raw = raw[raw.shape[0] // 2]
    return _to_pil(raw).convert("RGB")


def render_video(sample: dict) -> Image.Image:
    """Videos: extract the first frame as an RGB image."""
    video = sample.get("video")
    if video is None:
        return render_image_rgb(sample)

    # HuggingFace Video may decode to a list of PIL frames
    if isinstance(video, (list, tuple)) and len(video) > 0:
        return _to_pil(video[0]).convert("RGB")

    # Numpy array (T, H, W, C)
    if isinstance(video, np.ndarray):
        frame = video[0] if video.ndim == 4 else video
        return _to_pil(frame).convert("RGB")

    # torchcodec / custom decoder – try common attributes
    for attr in ("frames", "data"):
        if hasattr(video, attr):
            frames = getattr(video, attr)
            frame = frames[0] if hasattr(frames, "__getitem__") else frames
            if isinstance(frame, np.ndarray):
                return _to_pil(frame).convert("RGB")

    raise ValueError(f"Cannot extract a frame from video object {type(video)}")


MODALITY_RENDERERS: dict[str, callable] = {
    "image_rgb":     render_image_rgb,
    "image_gray":    render_image_gray,
    "image_stereo":  render_image_stereo,
    "image_binary":  render_image_binary,
    "image_medical": render_image_medical,
    "video":         render_video,
}

# ---------------------------------------------------------------------------
# Standardisation: resize + white-pad to a square canvas
# ---------------------------------------------------------------------------

def standardize(img: Image.Image, size: int = 224) -> Image.Image:
    """Resize *img* to *size* × *size*, scaling up small images with LANCZOS."""
    img = img.convert("RGB")
    return img.resize((size, size), Image.LANCZOS)

# ---------------------------------------------------------------------------
# Per-dataset loading + rendering
# ---------------------------------------------------------------------------

def _try_streaming_fallback(
    dataset_name: str,
    sample_idx: int,
    renderer,
    verbose: bool = False,
) -> Image.Image | None:
    """Try to stream a single sample from HuggingFace Hub as a fallback."""
    fallback = HF_STREAMING_FALLBACKS.get(dataset_name)
    if fallback is None:
        return None

    hub_id, image_key = fallback
    try:
        import datasets
        print(f"(streaming from {hub_id}) ", end="", flush=True)
        ds = datasets.load_dataset(hub_id, split="train", streaming=True)
        # Skip to sample_idx by consuming the iterator
        sample = next(itertools.islice(ds, sample_idx, sample_idx + 1))
        # Build a sample dict with "image" key for the renderer
        img = sample[image_key]
        return _to_pil(img).convert("RGB")
    except Exception as exc:
        print(f"  [STREAM-FAIL] {dataset_name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def load_and_render(
    dataset_name: str,
    sample_idx: int = 42,
    verbose: bool = False,
) -> Image.Image | None:
    """Return a raw (un-standardised) PIL image for *dataset_name*, or None."""
    try:
        from stable_datasets import images as sds_images
    except ImportError as exc:
        print(f"  [ERROR] Cannot import stable_datasets: {exc}", file=sys.stderr)
        return None

    dataset_cls = getattr(sds_images, dataset_name, None)
    if dataset_cls is None:
        # Not available locally — try HuggingFace streaming
        modality = DATASET_MODALITY.get(dataset_name, "image_rgb")
        renderer = MODALITY_RENDERERS[modality]
        result = _try_streaming_fallback(dataset_name, sample_idx, renderer, verbose)
        if result is not None:
            return result
        print(f"  [SKIP] {dataset_name}: not found in stable_datasets.images")
        return None

    extra_kwargs = DATASET_KWARGS.get(dataset_name, {})
    modality = DATASET_MODALITY.get(dataset_name, "image_rgb")
    renderer = MODALITY_RENDERERS[modality]

    # For very large datasets, prefer HF streaming to avoid huge downloads
    if dataset_name in PREFER_STREAMING:
        result = _try_streaming_fallback(dataset_name, sample_idx, renderer, verbose)
        if result is not None:
            return result

    # Try splits in priority order
    ds = None
    for split in ("train", "test", "validation"):
        try:
            ds = dataset_cls(split=split, **extra_kwargs)
            break
        except Exception:
            continue

    if ds is None:
        # Try HuggingFace Hub streaming as fallback
        result = _try_streaming_fallback(dataset_name, sample_idx, renderer, verbose)
        if result is not None:
            return result
        print(f"  [SKIP] {dataset_name}: could not load any split")
        return None

    try:
        # Streaming datasets are iterable-only (no len / __getitem__)
        if hasattr(ds, '__getitem__') and hasattr(ds, '__len__'):
            idx = sample_idx % len(ds)
            sample = ds[idx]
        else:
            sample = next(itertools.islice(ds, sample_idx, sample_idx + 1))
        return renderer(sample)
    except Exception as exc:
        print(f"  [FAIL] {dataset_name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        # Try streaming fallback on render failure too
        result = _try_streaming_fallback(dataset_name, sample_idx, renderer, verbose)
        if result is not None:
            return result
        return None

# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------

def generate_latex(
    rendered: dict[str, str],
    dataset_order: list[str],
    image_dir_name: str,
    ncols: int = 8,
) -> str:
    """Return a ``figure*`` LaTeX environment containing all rendered images.

    Args:
        rendered:       {dataset_name: saved_png_path} for successful renders.
        dataset_order:  Preferred ordering of dataset names.
        image_dir_name: Directory name to use in \\includegraphics paths.
        ncols:          Number of image columns in the grid.
    """
    available = [d for d in dataset_order if d in rendered]

    # Image width: share textwidth evenly, leaving a small margin for separators
    img_width = f"{0.92 / ncols:.4f}\\textwidth"

    # Split into rows of ncols
    rows: list[list[str]] = [
        available[i : i + ncols] for i in range(0, len(available), ncols)
    ]

    lines = [
        "% ─────────────────────────────────────────────────────────────────────",
        "% Magazine-style dataset samples — generated by create_magazine_image.py",
        "%",
        "% Required preamble:  \\usepackage{graphicx}",
        "% Upload the '" + image_dir_name + "/' directory alongside your .tex file.",
        "% Then include with:  \\input{" + image_dir_name + "/magazine_figure.tex}",
        "% ─────────────────────────────────────────────────────────────────────",
        "",
        "\\begin{figure*}[t]",
        "  \\centering",
        "  \\setlength{\\tabcolsep}{2pt}",
        "  \\renewcommand{\\arraystretch}{0.8}",
        f"  \\begin{{tabular}}{{{'c' * ncols}}}",
    ]

    for row in rows:
        # ---- image row ----
        img_cells = [
            f"\\includegraphics[width={img_width}]{{{image_dir_name}/{name}.png}}"
            for name in row
        ]
        lines.append("    " + " &\n    ".join(img_cells) + " \\\\[-2pt]")

        # ---- label row ----
        lbl_cells = [
            f"\\tiny {DATASET_LABELS.get(name, name)}"
            for name in row
        ]
        lines.append("    " + " &\n    ".join(lbl_cells) + " \\\\[4pt]")

    lines += [
        f"  \\end{{tabular}}",
        "  \\caption{%",
        "    One sample per dataset in the \\textsc{Stable-Datasets} benchmark suite.",
        "    Datasets span natural images, fine-grained recognition, synthetic",
        "    renderings, handwritten characters, medical scans, and video.}",
        "  \\label{fig:dataset-samples}",
        "\\end{figure*}",
    ]

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", default="dataset_samples",
        help="Directory for PNG images and the .tex file (default: dataset_samples/).",
    )
    parser.add_argument(
        "--ncols", type=int, default=0,
        help="Number of image columns in the LaTeX grid (default: auto = ceil(sqrt(n)) for a square grid).",
    )
    parser.add_argument(
        "--sample-idx", type=int, default=24,
        help="Sample index to draw from each dataset (default: 42).",
    )
    parser.add_argument(
        "--img-size", type=int, default=224,
        help="Side length in pixels for each saved PNG (default: 224).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full tracebacks for failed datasets.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered: dict[str, str] = {}
    n_total = len(DATASET_ORDER)

    for i, dataset_name in enumerate(DATASET_ORDER, 1):
        print(f"[{i:2d}/{n_total}] {dataset_name} ... ", end="", flush=True)
        raw_img = load_and_render(
            dataset_name, sample_idx=args.sample_idx, verbose=args.verbose
        )
        if raw_img is None:
            print("skipped")
            continue

        img = standardize(raw_img, size=args.img_size)
        save_path = out_dir / f"{dataset_name}.png"
        img.save(save_path)
        rendered[dataset_name] = str(save_path)
        print(f"saved → {save_path}")

    if not rendered:
        print(
            "\nNo datasets could be rendered. "
            "Make sure the datasets are downloaded and cached.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Auto-compute ncols for a square-ish grid: ceil(sqrt(n))
    ncols = args.ncols if args.ncols > 0 else math.ceil(math.sqrt(len(rendered)))

    # Write LaTeX figure
    latex = generate_latex(
        rendered=rendered,
        dataset_order=DATASET_ORDER,
        image_dir_name=args.output_dir,
        ncols=ncols,
    )
    tex_path = out_dir / "magazine_figure.tex"
    tex_path.write_text(latex)

    print(f"\n✓ Rendered {len(rendered)}/{n_total} datasets.")
    print(f"✓ LaTeX figure → {tex_path}")
    print()
    print("To use in Overleaf:")
    print(f"  1. Upload the '{args.output_dir}/' directory.")
    print(f"  2. Add \\usepackage{{graphicx}} to your preamble.")
    print(f"  3. \\input{{{args.output_dir}/magazine_figure.tex}}")


if __name__ == "__main__":
    main()
