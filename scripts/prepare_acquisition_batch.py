#!/usr/bin/env python3
"""Copy acquisition output folders into pipeline input batch layout."""

import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Acquisition output folder base")
    parser.add_argument("batch", help="Target batch name under data/input")
    parser.add_argument("--prefix", default="", help="Optional folder prefix filter")
    args = parser.parse_args()

    target_root = os.path.join("data", "input", args.batch)
    os.makedirs(target_root, exist_ok=True)

    copied = 0
    for name in sorted(os.listdir(args.source)):
        if args.prefix and not name.startswith(args.prefix):
            continue
        src = os.path.join(args.source, name)
        dst = os.path.join(target_root, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            copied += 1

    print(f"Copied {copied} acquisition folders into {target_root}")


if __name__ == "__main__":
    main()
