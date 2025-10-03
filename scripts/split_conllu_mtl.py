#!/usr/bin/env python3
import argparse
import random
import os

def read_conllu_file(filepath):
    """
    Reads the .conllu file and returns a list of sentences.
    Each sentence is defined as a block of text separated by a blank line.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    sentences = content.split("\n\n")
    return [s for s in sentences if s.strip()]

def write_conllu_file(sentences, filepath):
    """
    Writes a list of sentences to a .conllu file.
    Sentences are separated by double newlines.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(sentences))
        f.write("\n")

def write_markdown_report(report, filepath):
    """
    Writes the markdown report to the given filepath.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Markdown report written to: {filepath}")

def main():
    parser = argparse.ArgumentParser(
        description="Split a .conllu file for MTL: hold out a test set, then produce K disjoint train/dev splits."
    )
    parser.add_argument("input_file", help="Path to the input .conllu file")
    parser.add_argument("--num_seeds", type=int, default=3,
                        help="Number of seeds (folds) for MTL (default: 3)")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Proportion of sentences to hold out as test (default: 0.2)")
    parser.add_argument("--dev_ratio", type=float, default=0.33,
                        help="Within each fold, ratio of dev sentences (relative to the fold) (default: 0.33)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument("--out_dir", default="mtl_splits",
                        help="Output directory for the splits (default: mtl_splits)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Read and shuffle sentences
    sentences = read_conllu_file(args.input_file)
    if not sentences:
        raise ValueError("No sentences found in the input file.")
    random.seed(args.seed)
    random.shuffle(sentences)
    total_sentences = len(sentences)

    # First, hold out a test set from the whole data
    test_size = int(total_sentences * args.test_ratio)
    test_set = sentences[:test_size]
    remaining = sentences[test_size:]
    remaining_count = len(remaining)

    # For the remaining data, split into num_seeds disjoint folds.
    # For each seed, we will designate one fold as dev and the rest as train.
    fold_size = remaining_count // args.num_seeds
    folds = []
    start = 0
    for i in range(args.num_seeds):
        # Ensure the last fold takes any remainder
        if i == args.num_seeds - 1:
            folds.append(remaining[start:])
        else:
            folds.append(remaining[start:start+fold_size])
            start += fold_size

    # For each seed, create a train/dev split:
    for i in range(args.num_seeds):
        # Use fold i as dev, and all other folds as train
        dev_set = folds[i]
        train_set = [s for j, fold in enumerate(folds) if j != i for s in fold]

        # Optionally, you might want to further split train_set for dev if desired.
        # Here, we keep the dev set as the designated fold.
        seed_id = i + 1
        train_file = os.path.join(args.out_dir, f"seed{seed_id}_train.conllu")
        dev_file = os.path.join(args.out_dir, f"seed{seed_id}_dev.conllu")
        write_conllu_file(train_set, train_file)
        write_conllu_file(dev_set, dev_file)
        print(f"Seed {seed_id}: Train = {len(train_set)} sentences, Dev = {len(dev_set)} sentences.")

    # Write the held-out test set (common to all seeds)
    test_file = os.path.join(args.out_dir, "test.conllu")
    write_conllu_file(test_set, test_file)
    print(f"Test set: {len(test_set)} sentences written to {test_file}")

    # Create a markdown report summarizing the splits
    report = f"""# MTL Dataset Split Report

**Input File:** `{args.input_file}`
**Total Sentences:** {total_sentences}
**Test Ratio:** {args.test_ratio}  (Test set: {len(test_set)} sentences)
**Remaining Sentences:** {remaining_count}
**Number of Seeds/Folds:** {args.num_seeds}

For each seed, one fold is used as the dev set and the union of the other folds is used as the train set.

Detailed splits:
"""
    for i in range(args.num_seeds):
        seed_id = i + 1
        dev_count = len(folds[i])
        train_count = remaining_count - dev_count
        report += f"- **Seed {seed_id}:** Train = {train_count} sentences, Dev = {dev_count} sentences\n"

    report += f"\n**Held-out Test Set:** {len(test_set)} sentences (common for all seeds)\n"
    report += f"\n**Random Seed:** {args.seed}\n"
    report_path = os.path.join(args.out_dir, "dataset_split_report.md")
    write_markdown_report(report, report_path)

if __name__ == "__main__":
    main()
