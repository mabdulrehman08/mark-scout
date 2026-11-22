"""Find a reference mark in still images and export review evidence."""

import argparse
import hashlib
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageOps


DEFAULT_THRESHOLD = 0.30
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 30_000_000


def read_image(source):
    if urlsplit(source).scheme.lower() in {"http", "https"}:
        request = Request(source, headers={"User-Agent": "mark-scout/1.0"})
        with urlopen(request, timeout=15) as response:
            data = response.read(MAX_BYTES + 1)
    else:
        with Path(source).open("rb") as stream:
            data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("image exceeds 20 MiB")
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("image exceeds 30 megapixels")
        rgba = ImageOps.exif_transpose(image).convert("RGBA")
        matte = Image.new("RGBA", rgba.size, "white")
        rgb = np.asarray(Image.alpha_composite(matte, rgba).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), hashlib.sha256(data).hexdigest()


class LogoMatcher:
    def __init__(self, reference):
        self.sift = cv2.SIFT_create(nfeatures=3500)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.keypoints, self.descriptors = self.features(reference)
        self.height, self.width = reference.shape[:2]
        if len(self.keypoints) < 8:
            raise ValueError("reference needs at least 8 SIFT keypoints")

    def features(self, image):
        return self.sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)

    def locate(self, image):
        """Return a geometric candidate as (score, quad, inliers, tentative matches)."""
        scene_keys, scene_descriptors = self.features(image)
        if scene_descriptors is None or len(scene_descriptors) < 2:
            return 0.0, None, 0, 0

        # Keep the best correspondence for each scene keypoint.
        unique = {}
        for first, second in self.matcher.knnMatch(self.descriptors, scene_descriptors, k=2):
            if first.distance < 0.75 * second.distance:
                previous = unique.get(first.trainIdx)
                if previous is None or first.distance < previous.distance:
                    unique[first.trainIdx] = first
        matches = list(unique.values())
        if len(matches) < 8:
            return 0.0, None, 0, len(matches)

        ref_points = np.float32([self.keypoints[m.queryIdx].pt for m in matches])
        scene_points = np.float32([scene_keys[m.trainIdx].pt for m in matches])
        homography, mask = cv2.findHomography(ref_points, scene_points, cv2.RANSAC, 4.0)
        if homography is None or mask is None:
            return 0.0, None, 0, len(matches)
        inliers = mask.ravel().astype(bool)
        count = int(inliers.sum())
        if count < 8:
            return 0.0, None, count, len(matches)

        # Matches concentrated on one small detail cannot localize the full mark reliably.
        reference_area = (self.width - 1) * (self.height - 1)
        if cv2.contourArea(cv2.convexHull(ref_points[inliers])) < 0.04 * reference_area:
            return 0.0, None, count, len(matches)

        corners = np.float32([
            [0, 0], [self.width - 1, 0], [self.width - 1, self.height - 1],
            [0, self.height - 1],
        ])
        quad = cv2.perspectiveTransform(corners[None, :, :], homography)[0]
        height, width = image.shape[:2]
        if not valid_quad(quad, width, height):
            return 0.0, None, count, len(matches)

        score = (count / (count + 12)) * (count / len(matches))
        return score, quad, count, len(matches)


def valid_quad(quad, width, height):
    if not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
        return False
    if (np.abs(quad) > 3 * max(width, height)).any() or cv2.contourArea(quad, oriented=True) < 100:
        return False
    sides = np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)
    if sides.min() < 8 or sides.max() / sides.min() > 12:
        return False
    frame = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    visible, _ = cv2.intersectConvexConvex(quad, frame)
    return visible >= 0.45 * cv2.contourArea(quad)


def collect_sources(inputs, list_files, output_dir):
    sources = list(inputs)
    for name in list_files:
        entries = Path(name).read_text(encoding="utf-8-sig").splitlines()
        sources.extend(line.strip() for line in entries if line.strip() and not line.lstrip().startswith("#"))
    for source in sources:
        if urlsplit(source).scheme.lower() in {"http", "https"}:
            yield source
            continue
        path = Path(source)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
                    if not child.resolve().is_relative_to(output_dir.resolve()):
                        yield str(child)
        else:
            yield source


def annotate(image, quad, score, matched, target):
    if matched:
        cv2.polylines(image, [np.rint(quad).astype(np.int32)], True, (48, 224, 255), 3, cv2.LINE_AA)
    label = f"{'MATCH' if matched else 'NO MATCH'}  score {score:.3f}"
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(image, (0, 0), (text_width + 18, text_height + 18), (24, 24, 24), -1)
    cv2.putText(image, label, (9, text_height + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    encoded, buffer = cv2.imencode(".png", image)
    if not encoded:
        raise OSError("could not encode annotation")
    target.write_bytes(buffer.tobytes())


def scan(matcher, sources, output_dir, threshold):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, source in enumerate(sources, 1):
        row = {
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": None,
            "score": None,
            "box": None,
            "matched": None,
            "annotation": None,
        }
        try:
            image, row["sha256"] = read_image(source)
            score, quad, inliers, tentative = matcher.locate(image)
        except (OSError, ValueError, Image.DecompressionBombError, cv2.error) as exc:
            row["error"] = str(exc)
            print(f"ERROR     {source}: {exc}", file=sys.stderr)
        else:
            matched = quad is not None and score >= threshold
            row.update(
                score=round(score, 6), matched=matched, inliers=inliers, tentative_matches=tentative,
                box=[[round(float(x), 2), round(float(y), 2)] for x, y in quad] if matched else None,
            )
            target = output_dir / f"{index:04d}.png"
            try:
                annotate(image, quad, score, matched, target)
                row["annotation"] = target.name
            except (OSError, cv2.error) as exc:
                row["error"] = f"annotation: {exc}"
                print(f"ERROR     {source}: {row['error']}", file=sys.stderr)
            status = "MATCH" if matched else "NO MATCH"
            print(f"{status:<9} {score:.3f}  {source}")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="reference image path or HTTP(S) URL")
    parser.add_argument("sources", nargs="*", help="image paths, HTTP(S) URLs, or directories")
    parser.add_argument("--list", action="append", default=[], metavar="FILE",
                        help="read one source per line, with local paths relative to the working directory")
    parser.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_intermixed_args()
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    try:
        sources = list(collect_sources(args.sources, args.list, args.out))
        if not sources:
            parser.error("provide at least one image, URL, or nonempty directory")
        reference, reference_hash = read_image(args.reference)
        matcher = LogoMatcher(reference)
        rows = scan(matcher, sources, args.out, args.threshold)
        report = {
            "schema_version": 1,
            "reference": {"source": args.reference, "sha256": reference_hash},
            "threshold": args.threshold,
            "items": rows,
        }
        target = args.out / "evidence.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    except (OSError, ValueError, Image.DecompressionBombError, cv2.error) as exc:
        parser.exit(1, f"mark-scout: {exc}\n")
    print(f"Evidence: {target}")
    return int(any("error" in row for row in rows))


if __name__ == "__main__":
    sys.exit(main())
