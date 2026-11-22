"""Generate a seeded synthetic dataset, run the CLI, and score its JSON."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from mark_scout import DEFAULT_THRESHOLD


ROOT = Path(__file__).resolve().parent
SIZE = (360, 240)
WIDTH, HEIGHT = 720, 480
SEED = 24613
IOU_THRESHOLD = 0.50


def make_logo(decoy=False):
    """Draw an original badge with a ring, angular arch, and checker details."""
    logo = Image.new("RGBA", SIZE)
    draw = ImageDraw.Draw(logo)
    draw.rounded_rectangle((4, 4, 355, 235), radius=36, fill=(20, 36, 56), outline=(76, 197, 192), width=7)
    cream = (253, 233, 188)
    teal = (91, 220, 195)
    orange = (249, 161, 91)
    draw.arc((40, 25, 258, 215), 39 if decoy else 107, 322 if decoy else 400, fill=cream, width=9)
    if decoy:
        draw.polygon([(38, 51), (109, 174), (169, 91), (256, 187), (229, 201),
                      (167, 133), (106, 210), (18, 67)], fill=teal)
        draw.rectangle((119, 43, 145, 68), fill=orange)
    else:
        draw.polygon([(34, 174), (94, 64), (131, 109), (185, 36), (263, 175),
                      (233, 175), (182, 83), (132, 150), (98, 111), (65, 174)], fill=teal)
        draw.polygon([(149, 161), (171, 132), (198, 161), (171, 190)], fill=orange)
    for i, length in enumerate((45, 29, 39, 19) if decoy else (45, 19, 34, 27)):
        y = 74 + 24 * i
        draw.rounded_rectangle((281, y, 281 + length, y + 7), radius=3, fill=(cream, teal)[i % 2])
    for i in range(4):
        x = 77 + 21 * i
        draw.rectangle((x, 199 - 7 * (i % 2), x + 9, 208 - 7 * (i % 2)), fill=orange)
    return logo


def background(rng, style):
    base = [(130, 112, 89), (93, 111, 129), (177, 166, 149), (104, 116, 106)][style]
    x = np.linspace(-1, 1, WIDTH)[None, :]
    y = np.linspace(-1, 1, HEIGHT)[:, None]
    shade = 25 * x - 22 * y - 12 * (x * x + y * y)
    noise = rng.normal(0, 5, (HEIGHT, WIDTH))
    pixels = np.clip(np.array(base) + (shade + noise)[..., None], 0, 255).astype(np.uint8)
    scene = Image.fromarray(pixels)
    draw = ImageDraw.Draw(scene, "RGBA")
    if style == 0:
        for offset in range(45, HEIGHT + 85, 105):
            draw.line((0, offset - 52, WIDTH, offset + 40), fill=(40, 29, 21, 130), width=4)
            draw.line((0, offset - 48, WIDTH, offset + 44), fill=(247, 215, 177, 95), width=2)
    elif style == 1:
        for offset in range(-WIDTH, WIDTH, 68):
            draw.line((offset, 0, offset + 600, HEIGHT), fill=(205, 220, 230, 65), width=2)
    elif style == 2:
        for x0, y0 in [(12, 26), (420, -75), (300, 230)]:
            draw.rounded_rectangle((x0, y0, x0 + 300, y0 + 230),
                                   radius=12, fill=(235, 223, 199, 125), outline=(65, 58, 49, 95), width=3)
    else:
        for offset in range(40, WIDTH, 105):
            draw.rectangle((offset, 0, offset + 19, HEIGHT), fill=(42, 54, 50, 95))
    for _ in range(55):
        x0, y0 = (int(n) for n in rng.integers([0, 0], [WIDTH, HEIGHT]))
        radius = int(rng.integers(2, 33))
        color = tuple(int(n) for n in rng.integers(25, 235, 3)) + (int(rng.integers(43, 145)),)
        bounds = (x0, y0, x0 + radius * 2, y0 + radius)
        if rng.random() < 0.5:
            draw.ellipse(bounds, outline=color, width=3)
        else:
            draw.rectangle(bounds, fill=color)
    return np.asarray(scene).copy()


def place_badge(scene, logo, scale, angle, rng, occlude):
    badge = np.array(logo)
    rgb = badge[:, :, :3].astype(np.float32)
    gain = rng.uniform(0.70, 1.14, (1, 1, 3))
    shift = rng.uniform(-28, 31, (1, 1, 3))
    badge[:, :, :3] = np.clip(rgb * gain + shift, 0, 255).astype(np.uint8)
    width, height = SIZE
    center = ((width - 1) / 2, (height - 1) / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[:, 2] += rng.uniform([170, 150], [WIDTH - 170, HEIGHT - 150]) - center
    warped = cv2.warpAffine(badge, matrix, (WIDTH, HEIGHT))
    alpha = warped[:, :, 3:] / 255.0
    composite = np.rint(scene * (1 - alpha) + warped[:, :, :3] * alpha).astype(np.uint8)
    quad = cv2.transform(np.float32([[[0, 0], [width - 1, 0],
                                    [width - 1, height - 1], [0, height - 1]]]), matrix)[0]
    if occlude:
        # Cover one edge of the mark with a foreground strip.
        start, end = quad[0], quad[3]
        across = quad[1] - start
        down = end - start
        if rng.random() < 0.5:
            start, end = quad[1], quad[2]
            across *= -1
        coverage = rng.uniform(0.18, 0.42)
        occluder = np.float32([start - 0.11 * across - 0.17 * down,
                               start + coverage * across - 0.17 * down,
                               end + coverage * across + 0.12 * down,
                               end - 0.11 * across + 0.12 * down])
        color = tuple(int(n) for n in rng.integers(48, 215, 3))
        cv2.fillConvexPoly(composite, np.rint(occluder).astype(np.int32), color, cv2.LINE_AA)
    return composite, quad.tolist()


def generate():
    rng = np.random.default_rng(SEED)
    data = ROOT / "demo-data"
    (data / "scenes").mkdir(parents=True, exist_ok=True)
    logo = make_logo()
    decoy = make_logo(decoy=True)
    logo.save(data / "reference.png")
    truth = {}
    for positive in (True, False):
        kind = "positive" if positive else "negative"
        for index in range(24):
            filename = f"{kind}-{index + 1:02d}.jpg"
            scene = background(rng, index % 4)
            quad = None
            if positive:
                scale = rng.uniform(0.17, 0.36) if index % 4 == 3 else rng.uniform(0.38, 0.94)
                scene, quad = place_badge(scene, logo, scale, rng.uniform(-26, 26), rng, index % 3 == 1)
            elif index % 2 == 0:
                scene, _ = place_badge(scene, decoy, rng.uniform(0.35, 0.94),
                                       rng.uniform(-26, 26), rng, index % 3 == 1)
            Image.fromarray(scene).save(data / "scenes" / filename, quality=int(rng.integers(70, 96)))
            truth[filename] = quad
    (data / "truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="ascii")
    return truth


def quad_iou(box, truth):
    a, b = np.float32(box), np.float32(truth)
    intersection, _ = cv2.intersectConvexConvex(a, b)
    return intersection / (cv2.contourArea(a) + cv2.contourArea(b) - intersection)


def evaluate(truth, threshold):
    sources = [f"demo-data/scenes/{name}" for name in sorted(truth)]
    subprocess.run(
        [sys.executable, "mark_scout.py", "demo-data/reference.png", *sources,
         "--out", "results/eval", "--threshold", str(threshold)],
        cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True,
    )
    report = json.loads((ROOT / "results/eval/evidence.json").read_text(encoding="ascii"))
    rows = {Path(row["source"]).name: row for row in report["items"]}
    if len(report["items"]) != len(truth) or rows.keys() != truth.keys():
        raise RuntimeError("evidence report does not contain every evaluation image exactly once")
    good_ious = []
    tp = fp = tn = fn = 0
    for name, expected in truth.items():
        box = rows[name]["box"]
        if expected is None:
            fp += box is not None
            tn += box is None
        elif box is None:
            fn += 1
        else:
            iou = quad_iou(box, expected)
            if iou >= IOU_THRESHOLD:
                tp += 1
                good_ious.append(iou)
            else:
                # A bad localization is both a false alert and a missed target.
                fp += 1
                fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    mean_iou = float(np.mean(good_ious)) if good_ious else 0.0
    print(f"Synthetic evaluation (seed={SEED}, score>={threshold:.2f}, quad IoU>={IOU_THRESHOLD:.2f})")
    print(f"Images: {len(truth)} ({tp + fn} positive, {len(truth) - tp - fn} negative)")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Precision={precision:.3f} Recall={recall:.3f} Mean TP quad IoU={mean_iou:.3f}")
    print("Evidence: results/eval/evidence.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    evaluate(generate(), args.threshold)


if __name__ == "__main__":
    main()
