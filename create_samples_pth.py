#!/usr/bin/env python3
"""Create dataset/samples.pth from dataset/imgname.npy and contact_label_smplx.npy

Usage:
  python data/scripts/create_samples_pth.py --dataset-dir dataset --out dataset/samples.pth
"""
import argparse
import os
import sys
from typing import Any, List

import numpy as np
import torch


def load_npy(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path, allow_pickle=True)


def build_samples(imgnames: np.ndarray, labels: np.ndarray) -> List[dict]:
    if len(imgnames) != len(labels):
        raise ValueError(f"Length mismatch: imgname {len(imgnames)} vs labels {len(labels)}")
    samples = []
    for i in range(len(imgnames)):
        name = imgnames[i]
        # ensure string
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        else:
            name = str(name)

        verts = labels[i]
        try:
            verts_list = verts.tolist()
        except Exception:
            verts_list = verts

        samples.append({"imgname": name, "vertices": verts_list})

        if i==19:       # TODO
            break
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset", help="Path to dataset directory")
    parser.add_argument("--imgname", default=None, help="Path to imgname.npy (overrides dataset-dir)")
    parser.add_argument("--labels", default=None, help="Path to contact_label.npy/contact_label_smplx.npy (overrides dataset-dir)")
    parser.add_argument("--out", default=None, help="Output path for samples.pth")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    imgname_path = args.imgname or os.path.join(dataset_dir, "imgname.npy")
    labels_path = args.labels or os.path.join(dataset_dir, "contact_label.npy") # contact_label_smplx.npy
    out_path = args.out or os.path.join(dataset_dir, "samples_20.pth")      # TODO: samples.pth

    imgnames = load_npy(imgname_path)       # (4380,)
    labels = load_npy(labels_path)          # (4380,6890)

    samples = build_samples(imgnames, labels)

    # Ensure output dir exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    torch.save(samples, out_path)
    print(f"Saved {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
