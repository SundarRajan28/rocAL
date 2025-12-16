# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Test script for YOLO Label metadata reader.

This script validates that the YOLO Label metadata reader correctly parses
YOLO format .txt label files and converts them to pixel coordinates.

Usage:
    python yolo_label_reader.py --images <path_to_images> --labels <path_to_labels>

The script will:
1. Parse the YOLO .txt label files manually
2. Run the rocAL YOLO Label reader pipeline
3. Compare the outputs and report any discrepancies
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image


def parse_yolo_label_file(label_path, img_w, img_h):
    """Return list of (class_id, l, t, r, b) in pixels."""
    anns = []
    if not os.path.exists(label_path):
        return anns

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 5:
                continue

            cls = int(float(tokens[0]))

            if len(tokens) == 5:
                # Detection row: class xc yc w h (normalized)
                xc = float(tokens[1])
                yc = float(tokens[2])
                w = float(tokens[3])
                h = float(tokens[4])

                xc_px = xc * img_w
                yc_px = yc * img_h
                w_px = w * img_w
                h_px = h * img_h

                l = xc_px - w_px / 2.0
                t = yc_px - h_px / 2.0
                r = xc_px + w_px / 2.0
                b = yc_px + h_px / 2.0

                l = max(0.0, l)
                t = max(0.0, t)
                r = min(float(img_w), r)
                b = min(float(img_h), b)

                anns.append((cls, l, t, r, b))
            else:
                # Segmentation row: class x1 y1 x2 y2 ... xN yN (normalized)
                coords = [float(x) for x in tokens[1:]]
                if len(coords) < 4 or len(coords) % 2 != 0:
                    continue
                xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
                ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
                l = max(0.0, min(xs))
                r = min(float(img_w), max(xs))
                t = max(0.0, min(ys))
                b = min(float(img_h), max(ys))
                anns.append((cls, l, t, r, b))

    return anns


def build_ground_truth(images_dir, labels_dir):
    """Return dict: basename -> {'img_w', 'img_h', 'anns'}."""
    gt = {}
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    total = len(label_files)
    for idx, fname in enumerate(label_files, 1):
        stem = os.path.splitext(fname)[0]
        # YOLO reader supports JPEG only
        img_path = None
        for ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
            cand = os.path.join(images_dir, stem + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            print(f"No image found for label {fname}")
            continue
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        label_path = os.path.join(labels_dir, fname)
        anns = parse_yolo_label_file(label_path, img_w, img_h)
        gt[stem] = {"img_w": img_w, "img_h": img_h, "anns": anns}
        if total >= 10 and (idx % max(1, total // 10) == 0 or idx == total):
            print(f"Parsed annotations: {idx}/{total}")
    return gt


def run_rocal(images_dir, labels_dir, batch_size):
    """Run rocAL pipeline and return dict: basename -> [(cls, l, t, r, b), ...]."""
    try:
        from amd.rocal.pipeline import Pipeline
        import amd.rocal.fn as fn
        import amd.rocal.types as types
    except ImportError as e:
        print(f"Could not import rocAL: {e}")
        sys.exit(1)

    pipe = Pipeline(
        batch_size=batch_size,
        num_threads=1,
        device_id=0,
        seed=42,
        rocal_cpu=True,
        tensor_layout=types.NHWC,
        tensor_dtype=types.FLOAT,
    )

    with pipe:
        # New YOLO label reader; keep class ids as-is
        jpegs, bboxes, labels = fn.readers.yolo_label(
            labels_path=labels_dir,
            images_path=images_dir,
            ltrb=True,
            masks=False,
            avoid_class_remapping=True,
        )

        # Decode images so the pipeline has a tensor output
        images = fn.decoders.image(
            jpegs,
            file_root=images_dir,
            annotations_file=labels_dir,
            output_type=types.RGB,
            random_shuffle=False,
        )

        pipe.set_outputs(images)

    pipe.build()

    results = {}
    # Estimate total samples from labels
    num_labels = sum(1 for f in os.listdir(labels_dir) if f.endswith(".txt"))
    iters = (num_labels + batch_size - 1) // batch_size

    for it in range(iters):
        if pipe.rocal_run() != 0:
            raise StopIteration
        else:
            output_tensor_list = pipe.get_output_tensors()

        rocal_labels = pipe.get_bounding_box_labels()
        rocal_bboxes = pipe.get_bounding_box_cords()
        # Get image names for the current batch
        name_lens = np.empty(batch_size, dtype="int32")
        total_name_chars = pipe.get_image_name_length(name_lens)
        name_bytes = pipe.get_image_name(total_name_chars)

        # Split concatenated names using lengths
        names = []
        offset = 0
        for l in name_lens:
            if l <= 0:
                continue
            names.append(name_bytes[offset:offset + l].decode("utf-8"))
            offset += l

        for i, name in enumerate(names):
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            stem = os.path.splitext(os.path.basename(name))[0]

            img_labels = rocal_labels[i] if i < len(rocal_labels) else []
            flat_bboxes = rocal_bboxes[i] if i < len(rocal_bboxes) else []
            if len(flat_bboxes) > 0:
                arr = np.array(flat_bboxes, dtype=np.float32).reshape(-1, 4)
            else:
                arr = np.zeros((0, 4), dtype=np.float32)

            anns = []
            for j, cls in enumerate(img_labels):
                if j < arr.shape[0]:
                    l, t, r, b = arr[j]
                    anns.append((int(cls), float(l), float(t), float(r), float(b)))

            results[stem] = anns

        if iters >= 10 and (it + 1) % max(1, iters // 10) == 0:
            print(f"Processed rocAL batches: {it + 1}/{iters}")

    return results


def compare(gt, rocal, tol):
    ok = 0
    bad = 0
    total = len(gt)
    for img_idx, (stem, info) in enumerate(gt.items(), 1):
        gt_anns = info["anns"]
        if stem not in rocal:
            print(f"{stem} not in rocAL results")
            bad += 1
            continue
        ra_anns = rocal[stem]
        if len(gt_anns) != len(ra_anns):
            print(f"{stem}: expected {len(gt_anns)}, got {len(ra_anns)}")
            bad += 1
            continue
        all_match = True
        for ann_idx, (e, a) in enumerate(zip(gt_anns, ra_anns)):
            e_cls, e_l, e_t, e_r, e_b = e
            a_cls, a_l, a_t, a_r, a_b = a
            if e_cls != a_cls:
                print(f"{stem}[{ann_idx}]: expected {e_cls}, got {a_cls}")
                all_match = False
            if (
                abs(e_l - a_l) > tol
                or abs(e_t - a_t) > tol
                or abs(e_r - a_r) > tol
                or abs(e_b - a_b) > tol
            ):
                print(
                    f"{stem}[{ann_idx}]: "
                    f"expected ({e_l:.2f},{e_t:.2f},{e_r:.2f},{e_b:.2f}), "
                    f"got ({a_l:.2f},{a_t:.2f},{a_r:.2f},{a_b:.2f})"
                )
                all_match = False
        if all_match:
            ok += 1
        else:
            bad += 1
        if total >= 10 and (img_idx % max(1, total // 10) == 0 or img_idx == total):
            print(f"Compared images: {img_idx}/{total}")
    return ok, bad


def main():
    p = argparse.ArgumentParser("YOLO label metadata reader check")
    p.add_argument("--images", required=True, help="Path to JPEG images dir")
    p.add_argument("--labels", required=True, help="Path to YOLO .txt labels dir")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--tol", type=float, default=1.0, help="bbox tolerance in pixels")
    args = p.parse_args()

    print(f"Images: {args.images}")
    print(f"Labels: {args.labels}")

    gt = build_ground_truth(args.images, args.labels)
    print(f"Ground truth entries: {len(gt)}")

    rocal = run_rocal(args.images, args.labels, args.batch_size)
    print(f"rocAL entries: {len(rocal)}")

    ok, bad = compare(gt, rocal, args.tol)
    print(f"\nMatched images: {ok}, mismatches: {bad}")


if __name__ == "__main__":
    main()
