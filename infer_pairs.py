import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

from default_config import get_config
from src.slim import SLiM


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SLiM matching on arbitrary image pairs."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/home/disk1/Data/datasets/测试影像对"),
        help="Directory containing pair subdirectories or *_A/*_B images. Used only when image paths are not provided.",
    )
    parser.add_argument(
        "--image0_path",
        type=Path,
        default=None,
        help="Path to the first image.",
    )
    parser.add_argument(
        "--image1_path",
        type=Path,
        default=None,
        help="Path to the second image.",
    )
    parser.add_argument(
        "--pair_name",
        type=str,
        default=None,
        help="Output name for a single image pair. Defaults to image stem names.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/test_pairs"),
        help="Where to save match npz files and visualizations.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path("ckpt/megadepth_19epochs.ckpt"),
        help="Path to a SLiM checkpoint.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="outdoor_test",
        choices=["outdoor_test", "indoor_test"],
        help="Use outdoor_test for general/outdoor imagery, indoor_test for ScanNet-like imagery.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device, for example cuda:0.",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=None,
        help="Override both coarse and fine matching thresholds.",
    )
    parser.add_argument(
        "--refine_iters",
        type=int,
        default=4,
        help="Number of refinement iterations.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Resize longer image side before padding. Defaults to min(config IMAGE_SIZE, 960).",
    )
    parser.add_argument(
        "--max_vis_matches",
        type=int,
        default=300,
        help="Maximum matches drawn per pair.",
    )
    parser.add_argument(
        "--skip_warmup",
        action="store_true",
        help="Skip model.initial_forward(); useful if Triton warmup fails on a GPU.",
    )
    return parser.parse_args()


def find_image_pairs(input_dir: Path):
    files = [
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    pairs = []
    by_key = {}
    for path in files:
        stem = path.stem
        if stem.endswith("_A"):
            key = (path.parent, stem[:-2])
            by_key.setdefault(key, {})["A"] = path
        elif stem.endswith("_B"):
            key = (path.parent, stem[:-2])
            by_key.setdefault(key, {})["B"] = path
        else:
            continue

    for (parent, prefix), item in sorted(by_key.items()):
        if "A" in item and "B" in item:
            rel_parent = parent.relative_to(input_dir)
            name = str(rel_parent / prefix) if str(rel_parent) != "." else prefix
            pairs.append((name.replace("/", "__"), item["A"], item["B"]))
    return pairs


def safe_name(name: str):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def get_image_pairs(args):
    if args.image0_path is None and args.image1_path is None:
        pairs = find_image_pairs(args.input_dir)
        if not pairs:
            raise FileNotFoundError(
                f"No *_A / *_B image pairs found under {args.input_dir}"
            )
        return pairs

    if args.image0_path is None or args.image1_path is None:
        raise ValueError("--image0_path and --image1_path must be provided together.")

    path0 = args.image0_path
    path1 = args.image1_path
    for path in [path0, path1]:
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {path}")

    pair_name = args.pair_name or f"{path0.stem}__{path1.stem}"
    return [(safe_name(pair_name), path0, path1)]


def resize_and_pad_gray(path: Path, image_size: int, coarse_scale: int):
    image = Image.open(path).convert("L")
    orig_w, orig_h = image.size
    scale = image_size / max(orig_w, orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    image = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("L", (image_size, image_size), 0)
    canvas.paste(image, (0, 0))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr)[None]
    valid = torch.zeros((image_size, image_size), dtype=torch.bool)
    valid[:new_h, :new_w] = True
    coarse_mask = F.interpolate(
        valid[None, None].float(),
        size=(image_size // coarse_scale, image_size // coarse_scale),
        mode="nearest",
    )[0, 0].bool()

    # Coordinates produced by SLiM are multiplied by this scale to map back to
    # the original image before resizing.
    coord_scale = torch.tensor([orig_w / new_w, orig_h / new_h], dtype=torch.float32)
    return tensor, coarse_mask, coord_scale


def load_model(config, ckpt_path: Path, device: torch.device, skip_warmup: bool = False):
    model = SLiM(config=config["MODEL"])
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    if not skip_warmup:
        model.initial_forward()
    return model


def make_batch(path0: Path, path1: Path, image_size: int, coarse_scale: int, device):
    image0, mask0, scale0 = resize_and_pad_gray(path0, image_size, coarse_scale)
    image1, mask1, scale1 = resize_and_pad_gray(path1, image_size, coarse_scale)
    return {
        "image0": image0[None].to(device),
        "image1": image1[None].to(device),
        "mask0": mask0[None].to(device),
        "mask1": mask1[None].to(device),
        "scale0": scale0[None].to(device),
        "scale1": scale1[None].to(device),
        "pair_names": [(str(path0),), (str(path1),)],
    }


def draw_matches(path0: Path, path1: Path, pts0, pts1, out_path: Path, max_matches: int):
    img0 = Image.open(path0).convert("RGB")
    img1 = Image.open(path1).convert("RGB")
    w0, h0 = img0.size
    w1, h1 = img1.size
    canvas = Image.new("RGB", (w0 + w1, max(h0, h1)), "white")
    canvas.paste(img0, (0, 0))
    canvas.paste(img1, (w0, 0))
    draw = ImageDraw.Draw(canvas)

    n = len(pts0)
    if n > max_matches:
        idx = np.linspace(0, n - 1, max_matches).round().astype(np.int64)
        pts0 = pts0[idx]
        pts1 = pts1[idx]

    rng = np.random.default_rng(0)
    for p0, p1 in zip(pts0, pts1):
        color = tuple(int(c) for c in rng.integers(40, 230, size=3))
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]) + w0, float(p1[1])
        draw.line((x0, y0, x1, y1), fill=color, width=1)
        r = 2
        draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), outline=color, width=1)
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), outline=color, width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("SLiM inference in this repo uses CUDA timers; use a CUDA device.")
    torch.cuda.set_device(torch.device(args.device))

    config = get_config(args.config_name)
    config.MODEL.REFINE_ITERS = config.REFINE_ITERS = args.refine_iters
    if args.thr is not None:
        config.MODEL.COARSE_THRES = config.COARSE_THRES = args.thr
        config.MODEL.FINE_THRES = config.FINE_THRES = args.thr

    image_size = args.image_size or min(config.IMAGE_SIZE, 960)
    coarse_scale = int(config.MODEL.COARSE_SCALE)
    if image_size % coarse_scale != 0:
        raise ValueError(
            f"--image_size must be divisible by the coarse scale {coarse_scale}."
        )
    device = torch.device(args.device)
    pairs = get_image_pairs(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(config, args.ckpt_path, device, skip_warmup=args.skip_warmup)

    summary = []
    with torch.inference_mode():
        for name, path0, path1 in tqdm(pairs, desc="Matching image pairs"):
            batch = make_batch(path0, path1, image_size, coarse_scale, device)
            model(batch, training=False)

            pts0 = batch["fine_coord_0"].detach().cpu().numpy()
            pts1 = batch["fine_coord_1"].detach().cpu().numpy()
            npz_path = args.output_dir / f"{name}_matches.npz"
            vis_path = args.output_dir / f"{name}_matches.jpg"
            np.savez_compressed(
                npz_path,
                image0=str(path0),
                image1=str(path1),
                fine_coord_0=pts0,
                fine_coord_1=pts1,
            )
            draw_matches(path0, path1, pts0, pts1, vis_path, args.max_vis_matches)
            summary.append((name, len(pts0), npz_path, vis_path))
            del batch
            torch.cuda.empty_cache()

    print("\nSaved results:")
    for name, count, npz_path, vis_path in summary:
        print(f"{name}: {count} matches")
        print(f"  npz: {npz_path}")
        print(f"  vis: {vis_path}")


if __name__ == "__main__":
    main()
