from __future__ import print_function

import argparse
import io
import os

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merges all manifest CSV files in specified folder.')
    parser.add_argument('manifests',
                        nargs='+',
                        help='Path to all manifest files you want to merge')
    parser.add_argument('--output',
                        required=True,
                        help='Output path to merged manifest')

    args = parser.parse_args()

    file_paths = []
    for manifest in args.manifests:
        with open(manifest, 'r') as f:
            for line in f:
                file_paths.append(line)

    file_paths = sorted(file_paths, key=lambda x: float(x.split(',')[-1].strip()))
    with open(args.output, "w") as f:
        for line in file_paths:
            f.write(line)
