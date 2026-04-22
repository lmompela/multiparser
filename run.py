#!/usr/bin/env python3
"""
Multi-Task Parser for Creole Languages - Unified CLI

Supports:
- Baseline: Train on single language with multiple seeds
- Concatenated: Train on pre-combined datasets (hc+mc)
- Multi-task: Train on separate languages jointly (hcxmc)

Usage:
    python run.py train --mode baseline --language hc --model roberta
    python run.py train --mode concatenated --language hc+mc --model camembert
    python run.py train --mode multitask --languages hc,mc --model creoleval
    python run.py train --mode multitask --languages fr+hc,fr+mc --model creoleval --seeds 1
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


# Language configurations
LANGUAGE_CONFIG = {
    'hc': {
        'name': 'Haitian Creole',
        'data_dir': '../supar_creole/data/hc',
        'test': '../supar_creole/data/hc/hc_original_split_test.conllu'
    },
    'mc': {
        'name': 'Martinican Creole',
        'data_dir': '../supar_creole/data/mc',
        'test': '../supar_creole/data/mc/mc_original_split_test.conllu'
    },
    'fr': {
        'name': 'French',
        'data_dir': '../supar_creole/data/fr',
        'test': '../supar_creole/data/fr/fr_gsd-ud-test.conllu'
    },
    # Augmented configs: French + Creole combined (evaluate on Creole test only)
    'fr+hc': {
        'name': 'French + Haitian Creole',
        'data_dir': '../supar_creole/data/fr+hc',
        'test': '../supar_creole/data/hc/hc_original_split_test.conllu'
    },
    'fr+mc': {
        'name': 'French + Martinican Creole',
        'data_dir': '../supar_creole/data/fr+mc',
        'test': '../supar_creole/data/mc/mc_original_split_test.conllu'
    },
    'hc+mc': {
        'name': 'Haitian + Martinican (test on MC)',
        'data_dir': '../supar_creole/data/hc+mc',
        'test': '../supar_creole/data/mc/mc_original_split_test.conllu'
    },
    'mc+hc': {
        'name': 'Martinican + Haitian (test on HC)',
        'data_dir': '../supar_creole/data/mc+hc',
        'test': '../supar_creole/data/hc/hc_original_split_test.conllu'
    },
}

# Concatenated dataset configurations
CONCATENATED_CONFIG = {
    'hc+mc': {
        'name': 'Haitian + Martinican (concatenated)',
        'data_dir': '../supar_creole/data/hc+mc',
        'test': '../supar_creole/data/mc/mc_original_split_test.conllu'
    },
    'mc+hc': {
        'name': 'Martinican + Haitian (concatenated)',
        'data_dir': '../supar_creole/data/mc+hc',
        'test': '../supar_creole/data/hc/hc_original_split_test.conllu'
    },
    'fr+hc': {
        'name': 'French + Haitian (concatenated)',
        'data_dir': '../supar_creole/data/fr+hc',
        'test': '../supar_creole/data/hc/hc_original_split_test.conllu'
    },
    'fr+mc': {
        'name': 'French + Martinican (concatenated)',
        'data_dir': '../supar_creole/data/fr+mc',
        'test': '../supar_creole/data/mc/mc_original_split_test.conllu'
    },
}

# BERT model configurations
BERT_MODELS = {
    'roberta': 'roberta-large',
    'camembert': 'camembert-base',
    'camembert2': 'almanach/camembertv2-base',
    'creoleval': 'lgrobol/xlm-r-CreoleEval_all'
}

# Feature configurations: name → (encoder, feat_string)
# Mirrors supar_creole FEAT_CONFIGS for comparability.
FEAT_CONFIGS = {
    'bert-enc':           ('bert', ''),
    'lstm-tag-char-bert': ('lstm', 'tag,char,bert'),
    'lstm-tag-bert':      ('lstm', 'tag,bert'),
    'lstm-tag-char':      ('lstm', 'tag,char'),
    'lstm-tag':           ('lstm', 'tag'),
    'lstm-char':          ('lstm', 'char'),
    'lstm-bert':          ('lstm', 'bert'),
}

# Word embedding configurations (only meaningful when encoder=lstm)
# 'glove' uses SuPar's auto-download key (glove-6b-100), same as supar_creole.
EMBEDDINGS = {
    'glove':   ('embeddings/glove.6B.100d.txt', 100),
    'ht-ft':   ('embeddings/cc.ht.100.vec', 100),
    'fr-ft':   ('embeddings/cc.fr.100.vec', 100),   # place cc.fr.100.vec in embeddings/ (1.8G, not in git)
    'en-ft':   ('embeddings/cc.en.100.vec', 100),   # place cc.en.100.vec in embeddings/ (1.8G, not in git)
    'none':    ('', 100),
}


def get_seed_files(data_dir, seeds, split):
    """Get file paths for multiple seeds."""
    files = []
    for seed in seeds:
        file_path = f"{data_dir}/seed{seed}/{Path(data_dir).name}_{split}_seed{seed}.conllu"
        files.append(file_path)
    return files


def train_baseline(args):
    """Train baseline multi-task model (single language, multiple seeds)."""
    lang_config = LANGUAGE_CONFIG[args.language]
    bert_model = BERT_MODELS.get(args.model, args.model)
    encoder, feat = FEAT_CONFIGS[args.feat_config]
    embed_path, embed_dim = EMBEDDINGS[args.embed_key]
    embed_label = 'n-a' if encoder == 'bert' else args.embed_key

    # Naming mirrors supar_creole: {language}__{model}__{feat_config}__{embed}
    run_label = f"{args.language}__{args.model}__{args.feat_config}__{embed_label}"
    model_dir = Path(f"models/{run_label}")
    log_dir = Path(f"logs/{args.language}")

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training Baseline Multi-Task: {lang_config['name']}")
    print(f"Model: {args.model}  Feat: {args.feat_config}  Embed: {embed_label}")
    print(f"Seeds: {args.seeds}")
    print(f"{'='*60}\n")

    train_files = get_seed_files(lang_config['data_dir'], args.seeds, 'train')
    dev_files = get_seed_files(lang_config['data_dir'], args.seeds, 'dev')
    test_files = [lang_config['test']] * len(args.seeds)
    task_names = [f"{args.language}_seed{seed}" for seed in args.seeds]

    log_file = log_dir / f"{args.model}__{args.feat_config}__{embed_label}.log"

    config_file = 'config_bert_enc.ini' if encoder == 'bert' else args.config
    epochs = ['--epochs', '70'] if encoder == 'bert' else []

    cmd = [
        'python3', '-m', 'supar.cmds.multi_parser', 'train',
        '-b', '-d', str(args.device),
        '-c', config_file,
        '-p', str(model_dir) + '/',
        '--train', *train_files,
        '--dev', *dev_files,
        '--test', *test_files,
        '--task-names', *task_names,
        '--bert', bert_model,
        '--encoder', encoder,
        '--feat', feat,
        '--tree',
        '--seed', str(args.seeds[0]),
        *epochs,
    ]
    if encoder == 'lstm' and embed_path:
        cmd += ['--embed', embed_path]

    return run_with_logging(cmd, log_file if not args.no_log else None)


def train_concatenated(args):
    """Train on pre-concatenated datasets."""
    concat_config = CONCATENATED_CONFIG[args.language]
    bert_model = BERT_MODELS.get(args.model, args.model)
    encoder, feat = FEAT_CONFIGS[args.feat_config]
    embed_path, embed_dim = EMBEDDINGS[args.embed_key]
    embed_label = 'n-a' if encoder == 'bert' else args.embed_key

    run_label = f"{args.language}__{args.model}__{args.feat_config}__{embed_label}"
    model_dir = Path(f"models/{run_label}")
    log_dir = Path(f"logs/{args.language}")

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training Concatenated Multi-Task: {concat_config['name']}")
    print(f"Model: {args.model}  Feat: {args.feat_config}  Embed: {embed_label}")
    print(f"Seeds: {args.seeds}")
    print(f"{'='*60}\n")

    train_files = get_seed_files(concat_config['data_dir'], args.seeds, 'train')
    dev_files = get_seed_files(concat_config['data_dir'], args.seeds, 'dev')
    test_files = [concat_config['test']] * len(args.seeds)
    task_names = [f"{args.language}_seed{seed}" for seed in args.seeds]

    log_file = log_dir / f"{args.model}__{args.feat_config}__{embed_label}.log"

    config_file = 'config_bert_enc.ini' if encoder == 'bert' else args.config
    epochs = ['--epochs', '70'] if encoder == 'bert' else []

    cmd = [
        'python3', '-m', 'supar.cmds.multi_parser', 'train',
        '-b', '-d', str(args.device),
        '-c', config_file,
        '-p', str(model_dir) + '/',
        '--train', *train_files,
        '--dev', *dev_files,
        '--test', *test_files,
        '--task-names', *task_names,
        '--bert', bert_model,
        '--encoder', encoder,
        '--feat', feat,
        '--tree',
        '--seed', str(args.seeds[0]),
        *epochs,
    ]
    if encoder == 'lstm' and embed_path:
        cmd += ['--embed', embed_path]

    return run_with_logging(cmd, log_file if not args.no_log else None)


def train_multitask(args):
    """Train true multi-task model (separate languages, joint optimization)."""
    languages = args.languages.split(',')
    bert_model = BERT_MODELS.get(args.model, args.model)
    encoder, feat = FEAT_CONFIGS[args.feat_config]
    embed_path, embed_dim = EMBEDDINGS[args.embed_key]
    embed_label = 'n-a' if encoder == 'bert' else args.embed_key

    lang_combo = 'x'.join(languages)
    run_label = f"{lang_combo}__{args.model}__{args.feat_config}__{embed_label}"
    model_dir = Path(f"models/{run_label}")
    log_dir = Path(f"logs/{lang_combo}")

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training Multi-Task: {' + '.join([LANGUAGE_CONFIG[l]['name'] for l in languages])}")
    print(f"Model: {args.model}  Feat: {args.feat_config}  Embed: {embed_label}")
    print(f"Seeds: {args.seeds}")
    print(f"{'='*60}\n")

    train_files = []
    dev_files = []
    test_files = []
    task_names = []

    for lang in languages:
        lang_config = LANGUAGE_CONFIG[lang]
        train_files.extend(get_seed_files(lang_config['data_dir'], args.seeds, 'train'))
        dev_files.extend(get_seed_files(lang_config['data_dir'], args.seeds, 'dev'))
        test_files.extend([lang_config['test']] * len(args.seeds))
        task_names.extend([f"{lang}_seed{seed}" for seed in args.seeds])

    log_file = log_dir / f"{args.model}__{args.feat_config}__{embed_label}.log"

    config_file = 'config_bert_enc.ini' if encoder == 'bert' else args.config
    epochs = ['--epochs', '70'] if encoder == 'bert' else []

    cmd = [
        'python3', '-m', 'supar.cmds.multi_parser', 'train',
        '-b', '-d', str(args.device),
        '-c', config_file,
        '-p', str(model_dir) + '/',
        '--train', *train_files,
        '--dev', *dev_files,
        '--test', *test_files,
        '--task-names', *task_names,
        '--bert', bert_model,
        '--encoder', encoder,
        '--feat', feat,
        '--tree',
        '--seed', str(args.seeds[0]),
        *epochs,
    ]
    if encoder == 'lstm' and embed_path:
        cmd += ['--embed', embed_path]
    if getattr(args, 'joint_loss', False):
        cmd += ['--joint-loss']

    return run_with_logging(cmd, log_file if not args.no_log else None)


def predict(args):
    """Generate predictions using a trained model."""
    model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        sys.exit(1)
    
    pred_dir = Path(args.output).parent
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating predictions")
    print(f"Model: {model_path}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    cmd = [
        'python3', '-m', 'supar.cmds.multi_parser', 'predict',
        '-d', str(args.device),
        '-p', str(model_path),
        '--task', args.task,
        '--data', args.data,
        '--pred', args.output,
        '--tree'
    ]
    
    returncode = subprocess.run(cmd).returncode
    
    if returncode == 0:
        print(f"\n✓ Predictions saved to: {args.output}")
    else:
        print(f"\n✗ Prediction failed")
        sys.exit(1)


def evaluate(args):
    """Evaluate predictions against gold standard using external CoNLL eval script."""
    print(f"\n{'='*60}")
    print(f"Evaluating predictions")
    print(f"Gold: {args.gold}")
    print(f"Pred: {args.pred}")
    print(f"{'='*60}\n")

    eval_script = Path(__file__).parent / 'scripts' / 'conll17_ud_eval.py'

    cmd = ['python3', str(eval_script), '-v', args.gold, args.pred]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stderr)
        print(f"\n✗ Evaluation failed")
        sys.exit(1)

    print(result.stdout)

    # Parse UAS / LAS / CLAS from verbose output and save as JSON sidecar
    metrics = {}
    for line in result.stdout.splitlines():
        for key, label in [('UAS', 'UAS'), ('LAS', 'LAS'), ('CLAS', 'CLAS')]:
            if line.startswith(key):
                parts = line.split('|')
                if len(parts) >= 4:
                    try:
                        metrics[label] = round(float(parts[3].strip()), 2)
                    except ValueError:
                        pass

    json_path = Path(args.pred).with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump({
            'gold': args.gold,
            'pred': args.pred,
            **metrics
        }, f, indent=2)
    print(f"✓ Metrics saved to: {json_path}")


def run_with_logging(cmd, log_file=None):
    """Execute a command and optionally log output."""
    if log_file:
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in proc.stdout:
                line_str = line.decode()
                print(line_str, end='')
                f.write(line_str)
            proc.wait()
            return proc.returncode
    else:
        return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Task Parser - Unified CLI for Creole languages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a multi-task parser')
    train_parser.add_argument(
        '--mode',
        required=True,
        choices=['baseline', 'concatenated', 'multitask'],
        help='Training mode: baseline (single lang, multi-seed), concatenated (pre-combined data), multitask (separate langs, joint opt)'
    )
    train_parser.add_argument(
        '--language',
        help='Language code for baseline/concatenated mode (hc, mc, hc+mc, mc+hc)'
    )
    train_parser.add_argument(
        '--languages',
        help='Comma-separated languages for multitask mode (e.g., hc,mc)'
    )
    train_parser.add_argument(
        '--model', '-m',
        choices=list(BERT_MODELS.keys()),
        default='roberta',
        help='BERT model to use'
    )
    train_parser.add_argument(
        '--seeds',
        type=lambda s: [int(x) for x in s.split(',')],
        default='1,2,3',
        help='Random seeds (comma-separated)'
    )
    train_parser.add_argument(
        '--config', '-c',
        default='config2.ini',
        help='Config file path'
    )
    train_parser.add_argument(
        '--feat-config',
        choices=list(FEAT_CONFIGS.keys()),
        default='lstm-tag-char-bert',
        help='Feature/encoder configuration (mirrors supar_creole FEAT_CONFIGS)'
    )
    train_parser.add_argument(
        '--embed-key',
        choices=list(EMBEDDINGS.keys()),
        default='glove',
        help='Word embedding initialisation (ignored when feat-config=bert-enc)'
    )
    train_parser.add_argument(
        '--device', '-d',
        type=int,
        default=0,
        help='GPU device ID'
    )
    train_parser.add_argument(
        '--no-log',
        action='store_true',
        help='Do not save logs to file'
    )
    train_parser.add_argument(
        '--joint-loss',
        action='store_true',
        help='Use joint loss (sum losses across tasks per batch) instead of task-alternating. Recommended for Phase 4.1+ cross-language MTL.'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to trained model file (e.g. models/hc__creoleval__lstm-tag-char-bert__glove/hc_seed1.model)'
    )
    predict_parser.add_argument(
        '--task', '-t',
        required=True,
        help='Task name to use for prediction (must match a task head in the model, e.g. hc_seed1, fr+hc_seed2)'
    )
    predict_parser.add_argument(
        '--data', '-i',
        required=True,
        help='Input data file (CoNLL-U format)'
    )
    predict_parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output prediction file'
    )
    predict_parser.add_argument(
        '--device', '-d',
        type=int,
        default=0,
        help='GPU device ID'
    )
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate predictions')
    evaluate_parser.add_argument(
        '--gold', '-g',
        required=True,
        help='Gold standard data file'
    )
    evaluate_parser.add_argument(
        '--pred', '-p',
        required=True,
        help='Prediction file to evaluate'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'train':
        # Validate arguments based on mode
        if args.mode in ['baseline', 'concatenated']:
            if not args.language:
                print("✗ Error: --language required for baseline/concatenated mode")
                sys.exit(1)
            if args.mode == 'baseline' and args.language not in LANGUAGE_CONFIG:
                print(f"✗ Error: Invalid language '{args.language}' for baseline mode")
                print(f"   Valid options: {', '.join(LANGUAGE_CONFIG.keys())}")
                sys.exit(1)
            if args.mode == 'concatenated' and args.language not in CONCATENATED_CONFIG:
                print(f"✗ Error: Invalid language '{args.language}' for concatenated mode")
                print(f"   Valid options: {', '.join(CONCATENATED_CONFIG.keys())}")
                sys.exit(1)
        elif args.mode == 'multitask':
            if not args.languages:
                print("✗ Error: --languages required for multitask mode")
                sys.exit(1)
        
        # Call appropriate training function
        if args.mode == 'baseline':
            train_baseline(args)
        elif args.mode == 'concatenated':
            train_concatenated(args)
        elif args.mode == 'multitask':
            train_multitask(args)
    elif args.command == 'predict':
        predict(args)
    elif args.command == 'evaluate':
        evaluate(args)


if __name__ == '__main__':
    main()
