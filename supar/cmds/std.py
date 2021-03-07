# -*- coding: utf-8 -*-

import argparse

from supar.parsers.segmentation import Standardizer
from supar.cmds.cmd import parse


def main():
    parser = argparse.ArgumentParser(description='Create Biaffine Dependency Parser.')
    parser.set_defaults(Parser=Standardizer)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', default='tag,char', help='additional features to use，separated by commas.')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default='data/stdseg/train.conll', help='path to train file')
    subparser.add_argument('--dev', default='data/stdseg/dev.conll', help='path to dev file')
    subparser.add_argument('--test', default='data/stdseg/test.conll', help='path to test file')
    subparser.add_argument('--embed', default=None, help='path to pretrained embeddings')
    subparser.add_argument('--unk', default='unk', help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed', default=100, type=int, help='dimension of embeddings')
    subparser.add_argument('--bert', default='bert-base-cased', help='which bert model to use')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/stdseg/test.conll', help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/stdseg/test.conll', help='path to dataset')
    subparser.add_argument('--pred', default='exp/seg/1/pred.conll', help='path to predicted result')
    parse(parser)


if __name__ == "__main__":
    main()
