"""
Aggregate evaluation results across seeds for all configs in a predictions directory.
Reads pre-saved .json files produced by `run.py evaluate` (one per prediction file).

Directory layout (multiparser-specific):
  predictions/{config}/{lang}_seed{n}.json
  e.g. predictions/hc__creoleval__bert-enc__n-a/hc_seed1.json

Usage:
  # Aggregate all configs for a language (HC or MC):
  python3 scripts/aggregate_results.py --lang hc
  python3 scripts/aggregate_results.py --lang mc

  # Single config, all seeds (quick check):
  python3 scripts/aggregate_results.py --lang hc --config "hc__creoleval__bert-enc__n-a"

  # Save output to a file (shows per-seed detail + averages):
  python3 scripts/aggregate_results.py --lang hc --seeds 1,2,3 --output results/phase4_hc.txt
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GOLD_FILES = {
    'hc': '../supar_creole/data/hc/hc_original_split_test.conllu',
    'mc': '../supar_creole/data/mc/mc_original_split_test.conllu',
}


def fmt_mean(vals):
    if not vals:
        return '     —     '
    if len(vals) == 1:
        return f'   {vals[0]:6.2f}    '
    mean = statistics.mean(vals)
    std = statistics.stdev(vals)
    return f'{mean:5.2f} ± {std:.2f}'


def build_report(lang, configs, gold_file):
    lines = []
    W = 92

    lines.append('')
    lines.append('=' * W)
    lines.append(f'  PER-SEED DETAIL — {lang.upper()} | Gold: {gold_file}')
    lines.append('=' * W)

    sorted_configs = sorted(
        configs.items(),
        key=lambda x: -statistics.mean(v.get('LAS', 0) for v in x[1].values())
    )

    for config, seed_data in sorted_configs:
        seeds = sorted(seed_data.keys())
        lines.append('')
        lines.append(f'  Config: {config}')
        lines.append(f"  {'':>10}  {'UAS':>8}  {'LAS':>8}  {'CLAS':>8}")
        lines.append(f"  {'-'*40}")

        uas_vals, las_vals, clas_vals = [], [], []
        for s in seeds:
            m = seed_data[s]
            uas  = m.get('UAS',  None)
            las  = m.get('LAS',  None)
            clas = m.get('CLAS', None)
            uas_str  = f'{uas:8.2f}'  if uas  is not None else '       —'
            las_str  = f'{las:8.2f}'  if las  is not None else '       —'
            clas_str = f'{clas:8.2f}' if clas is not None else '       —'
            lines.append(f"  {'seed ' + str(s):>10}  {uas_str}  {las_str}  {clas_str}")
            if uas  is not None: uas_vals.append(uas)
            if las  is not None: las_vals.append(las)
            if clas is not None: clas_vals.append(clas)

        if len(seeds) > 1:
            lines.append(f"  {'-'*40}")
            def _avg(vals):
                return f'{statistics.mean(vals):8.2f}' if vals else '       —'
            def _std(vals):
                return f'± {statistics.stdev(vals):.2f}' if len(vals) > 1 else ''
            lines.append(
                f"  {'mean':>10}  {_avg(uas_vals)}  {_avg(las_vals)}  {_avg(clas_vals)}"
            )
            lines.append(
                f"  {'std':>10}  {_std(uas_vals):>8}  {_std(las_vals):>8}  {_std(clas_vals):>8}"
            )

    lines.append('')
    lines.append('=' * W)
    lines.append(f'  SUMMARY TABLE — {lang.upper()} | Gold: {gold_file}')
    lines.append('=' * W)
    lines.append(f"{'Config':<55} {'Seeds':>8}  {'UAS':>13}  {'LAS':>13}  {'CLAS':>13}")
    lines.append('-' * W)

    for config, seed_data in sorted_configs:
        seeds = sorted(seed_data.keys())
        seed_str  = '+'.join(str(s) for s in seeds)
        uas_vals  = [seed_data[s]['UAS']  for s in seeds if 'UAS'  in seed_data[s]]
        las_vals  = [seed_data[s]['LAS']  for s in seeds if 'LAS'  in seed_data[s]]
        clas_vals = [seed_data[s]['CLAS'] for s in seeds if 'CLAS' in seed_data[s]]
        lines.append(
            f'{config:<55} {seed_str:>8}  {fmt_mean(uas_vals):>13}'
            f'  {fmt_mean(las_vals):>13}  {fmt_mean(clas_vals):>13}'
        )

    lines.append('=' * W)
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Aggregate multi-seed evaluation results.')
    parser.add_argument('--lang', '-l', required=True, choices=['hc', 'mc'],
                        help='Language: hc or mc')
    parser.add_argument('--config', '-c', default=None,
                        help='Filter to a specific config name (substring match)')
    parser.add_argument('--pred-dir', default=None,
                        help='Override predictions root directory (default: predictions/)')
    parser.add_argument('--seeds', default=None,
                        help='Only include these seeds (comma-separated, e.g. 1,2,3)')
    parser.add_argument('--output', '-o', default=None,
                        help='Save report to this file (also printed to stdout)')
    args = parser.parse_args()

    pred_root = Path(args.pred_dir or 'predictions')
    seed_filter = set(int(s) for s in args.seeds.split(',')) if args.seeds else None

    if not pred_root.exists():
        print(f'Error: predictions directory not found: {pred_root}', file=sys.stderr)
        sys.exit(1)

    # Multiparser layout: predictions/{config}/{lang}_seed{n}.json
    # Config subdirs begin with the language prefix.
    configs = defaultdict(dict)
    missing_json = []

    for config_dir in sorted(pred_root.iterdir()):
        if not config_dir.is_dir():
            continue
        if not config_dir.name.startswith(f'{args.lang}__'):
            continue
        if args.config and args.config not in config_dir.name:
            continue

        config_name = config_dir.name

        for json_file in sorted(config_dir.glob(f'{args.lang}_seed*.json')):
            stem = json_file.stem  # e.g. hc_seed1
            try:
                seed = int(stem.split('_seed')[1])
            except (IndexError, ValueError):
                continue
            if seed_filter and seed not in seed_filter:
                continue
            with open(json_file) as f:
                data = json.load(f)
            configs[config_name][seed] = {
                k: data[k] for k in ('UAS', 'LAS', 'CLAS') if k in data
            }

        for pred_file in sorted(config_dir.glob(f'{args.lang}_seed*.conllu')):
            stem = pred_file.stem
            try:
                seed = int(stem.split('_seed')[1])
            except (IndexError, ValueError):
                continue
            if seed_filter and seed not in seed_filter:
                continue
            if not pred_file.with_suffix('.json').exists():
                missing_json.append(str(pred_file))

    if missing_json:
        print("⚠  No JSON found for (run `python3 run.py evaluate` on these first):")
        for f in missing_json:
            print(f"   {f}")
        print()

    if not configs:
        print(f'No result JSON files found under {pred_root}/ for language={args.lang}')
        sys.exit(0)

    report = build_report(args.lang, configs, GOLD_FILES[args.lang])
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f'Report saved to: {out_path}')


if __name__ == '__main__':
    main()
