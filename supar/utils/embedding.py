# -*- coding: utf-8 -*-

import torch


class Embedding(object):

    def __init__(self, tokens, vectors, unk=None):
        self.tokens = tokens
        self.vectors = torch.tensor(vectors)
        self.pretrained = {w: v for w, v in zip(tokens, vectors)}
        self.unk = unk

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, token):
        return token in self.pretrained

    @property
    def dim(self):
        return self.vectors.size(1)

    @property
    def unk_index(self):
        if self.unk is not None:
            return self.tokens.index(self.unk)
        else:
            raise AttributeError

    @classmethod
    def load(cls, path, unk=None):
        with open(path, 'r') as f:
            lines = [line for line in f]

        # Auto-detect and skip fastText header line (format: "vocab_size dim")
        start = 0
        first = lines[0].strip().split()
        if len(first) == 2 and all(tok.isdigit() for tok in first):
            start = 1

        splits = [line.split() for line in lines[start:]]
        tokens, vectors = zip(*[(s[0], list(map(float, s[1:])))
                                 for s in splits if len(s) > 1])

        return cls(tokens, vectors, unk=unk)
