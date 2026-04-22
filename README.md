# Multi-Task Parser for Creole Languages

Unified CLI for multi-task dependency parsing experiments (fork/extension of SuPar-style biaffine parsing with task-specific heads and a shared encoder).

## Training modes

1. `baseline`: one language, multiple seeds treated as separate tasks
2. `concatenated`: pre-combined datasets per seed (e.g., `hc+mc`)
3. `multitask`: separate language datasets trained jointly (e.g., `hc,mc` or `fr+hc,fr+mc`)

## Supported language/task codes

- `hc`, `mc`, `fr`
- `hc+mc`, `mc+hc`
- `fr+hc`, `fr+mc`

## Supported model keys

- `roberta`
- `camembert`
- `camembert2`
- `creoleval`

## Installation

```bash
git clone git@github.com:lmompela/multiparser.git
cd multiparser
source ../supar_creole/supar_venv/bin/activate
pip install -e .
```

If needed, create your own venv and install dependencies as usual.

## Embeddings

For LSTM feature configs, embeddings are expected under `embeddings/`:

- `glove` -> `embeddings/glove.6B.100d.txt`
- `ht-ft` -> `embeddings/cc.ht.100.vec`
- `fr-ft` -> `embeddings/cc.fr.100.vec`
- `en-ft` -> `embeddings/cc.en.100.vec`
- `none` -> random init

For `bert-enc`, `--embed-key` is ignored.

## Quick start

### Baseline (single language, multi-seed tasks)

```bash
python run.py train \
  --mode baseline \
  --language hc \
  --model creoleval \
  --feat-config lstm-tag-char-bert \
  --embed-key glove \
  --seeds 1,2,3
```

### Concatenated

```bash
python run.py train \
  --mode concatenated \
  --language hc+mc \
  --model creoleval \
  --feat-config lstm-tag-char-bert \
  --embed-key glove \
  --seeds 1,2,3
```

### True multitask (shared encoder across tasks)

```bash
python run.py train \
  --mode multitask \
  --languages fr+hc,fr+mc \
  --model creoleval \
  --feat-config bert-enc \
  --joint-loss \
  --seeds 1,2,3
```

## CLI reference

### `train`

```bash
python run.py train --mode {baseline,concatenated,multitask} [OPTIONS]
```

Key options:

- `--language` for `baseline|concatenated`
- `--languages` for `multitask` (comma-separated)
- `--model {roberta,camembert,camembert2,creoleval}`
- `--feat-config {bert-enc,lstm-tag-char-bert,lstm-tag-bert,lstm-tag-char,lstm-tag,lstm-char,lstm-bert}`
- `--embed-key {glove,ht-ft,fr-ft,en-ft,none}`
- `--seeds 1,2,3`
- `--config` (default `config2.ini`; auto-switches to `config_bert_enc.ini` for `bert-enc`)
- `--joint-loss` (recommended for cross-language multitask experiments)
- `--device`, `--no-log`

### `predict`

`predict` requires a model path, a **task name**, and input/output files:

```bash
python run.py predict \
  --model models/fr+hcxfr+mc__creoleval__bert-enc__n-a/fr+hc_seed1.model \
  --task fr+hc_seed1 \
  --data ../supar_creole/data/hc/hc_original_split_test.conllu \
  --output predictions/fr+hc_seed1.conllu
```

### `evaluate`

```bash
python run.py evaluate \
  --gold ../supar_creole/data/hc/hc_original_split_test.conllu \
  --pred predictions/fr+hc_seed1.conllu
```

This runs `scripts/conll17_ud_eval.py` and writes a JSON metrics sidecar next to prediction output.

## Output naming convention

- Model directory: `models/{lang_combo}__{model}__{feat_config}__{embed_label}/`
- Log file: `logs/{lang_combo}/{model}__{feat_config}__{embed_label}.log`
- Predictions: `predictions/{lang_combo}__{model}__{feat_config}__{embed_label}/{task}.conllu`

Where `lang_combo` is `x`-joined for multitask (e.g., `fr+hcxfr+mc`) and `embed_label` is `n-a` for `bert-enc`.

## License

MIT License
