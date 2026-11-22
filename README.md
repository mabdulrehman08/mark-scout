# mark-scout

mark-scout is a Python CLI for finding a reference logo in still images. It
accepts local image files, HTTP(S) image URLs, and folders. It draws a bounding
quadrilateral around the best match per image and writes a JSON report that a
reviewer can inspect alongside the annotated copies.

OpenCV SIFT finds local features in grayscale. A 0.75 ratio test rejects
ambiguous descriptors, then RANSAC fits a homography from the reference to the
image. The score multiplies the inlier fraction by `inliers / (inliers + 12)`
to favor fits with more supporting points. A match also needs spatially spread
inliers and a plausible projected quad. The score is a ranking heuristic.

## Quickstart

```sh
python -m pip install -r requirements.txt
python evaluate.py
python mark_scout.py demo-data/reference.png demo-data/scenes/positive-01.jpg demo-data/scenes/negative-02.jpg --out results/quickstart
```

`evaluate.py` generates the reference, scenes, and ground-truth quads in
`demo-data/`. To scan a folder recursively, pass `demo-data/scenes` in place of
the two image paths. You can also pass image URLs directly or add
`--list sources.txt` with one file, folder, or URL per line. Relative paths in
the list resolve from the current working directory. Use `--threshold` to
override the default 0.30 score cutoff.

The output directory contains a numbered PNG for each decoded image and an
`evidence.json` report. The report records the source, UTC processing timestamp,
SHA-256 of the bytes read, score, and detection result. A match includes four
`[x, y]` corners in oriented image pixels, in reference corner order starting
at the top left. A failed input gets an error entry and the rest of the batch
continues. The CLI exits with status 1 if any input fails.

## Actual run

Python 3.13.1, OpenCV 4.14.0, NumPy 2.5.1, and Pillow 10.4.0 on Windows.
The two-image quickstart printed:

```text
MATCH     0.817  demo-data/scenes/positive-01.jpg
NO MATCH  0.000  demo-data/scenes/negative-02.jpg
Evidence: results\quickstart\evidence.json
```

The matching item in that run's evidence report:

```json
{
  "source": "demo-data/scenes/positive-01.jpg",
  "timestamp": "2026-11-22T18:16:02+00:00",
  "sha256": "e9f47b2ac03bbde426200f4d064b53f33767f86da6452dc4833594ef428159ea",
  "score": 0.816587,
  "box": [[298.14, 91.14], [600.37, 231.51], [507.13, 432.69], [204.58, 292.25]],
  "matched": true,
  "annotation": "0001.png",
  "inliers": 64,
  "tentative_matches": 66
}
```

## Synthetic evaluation

The evaluator draws all artwork and backgrounds programmatically. It inserts
an original badge into 24 JPEG scenes with randomized scale, rotation,
position, color, compression, and partial occlusion in eight scenes. The 24
negatives include 12 scenes with a similar decoy badge. A true positive needs
at least 0.50 quad intersection over union (IoU) with the known placement.
An alert with a misplaced quad counts as both a false positive and a missed
target.

```text
Synthetic evaluation (seed=24613, score>=0.30, quad IoU>=0.50)
Images: 48 (24 positive, 24 negative)
TP=24 FP=1 TN=23 FN=0
Precision=0.960 Recall=1.000 Mean TP quad IoU=0.985
Evidence: results/eval/evidence.json
```

The false positive was the decoy in `negative-03.jpg` at score 0.324675.
These metrics describe one generated badge on a small synthetic set.

## Limitations

Classical features can miss tiny or low-texture logos, severe occlusions,
and recolorings that erase grayscale contrast. Similar artwork can
produce a convincing fit. The CLI reports one candidate per image, and
remote URLs must return image bytes.

A production system could use a learned detector such as YOLO trained on
synthetic composites and real labeled examples, embedding retrieval to
shortlist candidates at web scale, and human review to decide which alerts
warrant follow-up.

I'm Muhammad Abdulrehman, a Drexel CS student who trains YOLO and RF-DETR detectors in Python and labels data in Roboflow for a robot-arm vision lab.
