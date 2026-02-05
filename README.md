# Multi-Task Parser for Creole Languages

This repository provides a unified CLI for training multi-task dependency parsers for Creole languages. It's a fork of the [Multi-Task Parser](https://github.com/Jbar-ry/Multi-Task-Parser) which extends [SuPar](https://github.com/yzhangcs/parser).

## Features

**Three Training Modes:**

1. **Baseline**: Single language with multiple seeds (multi-task across seeds)
2. **Concatenated**: Pre-combined datasets (e.g., HC+MC concatenated before training)
3. **Multi-Task**: Separate languages with joint optimization (true multi-task learning)

**Supported Languages:**
- Haitian Creole (hc)
- Martinican Creole (mc)
- French (fr)

**Supported Models:**
- RoBERTa Large
- CamemBERT Base
- CamemBERT v2 Base
- XLM-R CreoleEval

## Installation

### 1. Clone the repository
```bash
git clone git@github.com:lmompela/multiparser.git
cd multiparser
```

### 2. Install dependencies
```bash
# Use existing virtual environment from supar_creole
source ../supar_creole/supar_venv/bin/activate

# Or create new one
python3 -m venv venv
source venv/bin/activate
pip install -U supar
```

### 3. Get embeddings (103MB, not in repo)
```bash
# Copy from supar_creole if available
cp ../supar_creole/embeddings/cc.ht.100.vec embeddings/

# Or download FastText embeddings
cd embeddings
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ht.300.bin.gz
gunzip cc.ht.300.bin.gz
```

## Quick Start

### Baseline Mode (Single Language, Multi-Seed)

Train on Haitian Creole with 3 seeds:
```bash
python run.py train \
  --mode baseline \
  --language hc \
  --model roberta \
  --seeds 1,2,3
```

### Concatenated Mode (Pre-Combined Data)

Train on concatenated HC+MC dataset:
```bash
python run.py train \
  --mode concatenated \
  --language hc+mc \
  --model roberta \
  --seeds 1,2,3
```

### Multi-Task Mode (Joint Optimization)

Train on HC and MC jointly (6 tasks: 3 seeds × 2 languages):
```bash
python run.py train \
  --mode multitask \
  --languages hc,mc \
  --model creoleval \
  --seeds 1,2,3
```

## Understanding Training Modes

### Baseline Mode
- **What**: Train on single language with multiple random seeds
- **How**: Each seed is treated as a separate task
- **Example**: 3 HC seeds → 3 tasks (`hc_seed1`, `hc_seed2`, `hc_seed3`)

### Concatenated Mode
- **What**: Train on pre-combined datasets
- **How**: Datasets are concatenated **before** training
- **Example**: HC+MC concatenated data → 3 tasks (one per seed of combined data)
- **Data location**: `../supar_creole/data/hc+mc/`

### Multi-Task Mode
- **What**: True multi-task learning with separate treebanks
- **How**: Each language×seed is a separate task, joint optimization
- **Example**: HC,MC with 3 seeds → 6 tasks (`hc_seed1-3`, `mc_seed1-3`)

## Usage

### Training
```bash
python run.py train --mode {baseline|concatenated|multitask} [OPTIONS]
```

**Examples:**
```bash
# Baseline
python run.py train --mode baseline --language hc --model roberta

# Concatenated
python run.py train --mode concatenated --language hc+mc --model camembert

# Multi-task
python run.py train --mode multitask --languages hc,mc --model creoleval
```

### Prediction
```bash
python run.py predict \
  --model models/mtl_hc_baseline/roberta \
  --data test.conllu \
  --output predictions/test_pred.conllu
```

### Evaluation
```bash
python run.py evaluate \
  --gold gold.conllu \
  --pred predictions/test_pred.conllu
```

## Migration from Shell Scripts

**Old way:**
```bash
./run_training_hc_baseline_enhanced.sh
./run_training_hc+mc_enhanced.sh
./run_training_mcxhc_enhanced.sh
```

**New way:**
```bash
python run.py train --mode baseline --language hc
python run.py train --mode concatenated --language hc+mc
python run.py train --mode multitask --languages mc,hc
```

## Citation

```bibtex
@InProceedings{alisayyed:dakota:2021,
  author    = {Ali Syed, Hatem and Dakota, Daniel and Kübler, Sandra},
  title     = {Exploring Constituency and Multi-Task Parsers for Low-Resource Languages},
  booktitle = {Proceedings of the 17th International Conference on Parsing Technologies (IWPT)},
  year      = {2021},
  pages     = {54--60}
}
```

## License

MIT License

For detailed documentation, see `python run.py --help` or [README.backup.md](README.backup.md) for upstream documentation.
