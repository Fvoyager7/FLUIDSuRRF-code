#!/usr/bin/env python3
"""Run detect_lakes.py locally for Greenland SW granules listed in a CSV file."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--granule-list', default='granule_lists/GrIS_2022_GRE_2000_SW.csv')
    parser.add_argument('--max-jobs', type=int, default=1, help='Number of granules to process')
    parser.add_argument('--start-index', type=int, default=0, help='Start row in the granule list')
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()

    list_path = args.granule_list if os.path.isabs(args.granule_list) else os.path.join(REPO_ROOT, args.granule_list)
    detect_script = os.path.join(REPO_ROOT, 'detect_lakes.py')

    with open(list_path, encoding='utf-8') as handle:
        rows = [line.strip().split(',') for line in handle if line.strip()]

    selected = rows[args.start_index: args.start_index + args.max_jobs]
    if not selected:
        raise SystemExit('No granules selected. Check --granule-list / --start-index / --max-jobs.')

    for granule, polygon, *_rest in selected:
        cmd = [
            args.python,
            detect_script,
            '--granule', granule,
            '--polygon', polygon,
        ]
        print('\n>>>', ' '.join(cmd))
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    print('\nCompleted', len(selected), 'granule(s). Check detection_out_data/, detection_out_plot/, detection_out_stat/.')


if __name__ == '__main__':
    main()
