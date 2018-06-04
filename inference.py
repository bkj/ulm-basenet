#!/usr/bin/env python

"""
    inference.py
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from functools import partial

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from basenet.helpers import set_seeds, set_freeze, to_numpy
from basenet.text.data import RaggedDataset, text_collate_fn

from ulmfit import TextClassifier, basenet_train

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm-weights-path', type=str)
    parser.add_argument('--X', type=str)
    parser.add_argument('--outpath', type=str, default='preds')
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

# --
# Run

if __name__ == "__main__":
    # --
    # Params
    
    bptt, emb_sz, n_hid, n_layers, batch_size = 70, 400, 1150, 3, 48
    max_seq = 20 * 70
    pad_token = 1
    
    args = parse_args()
    set_seeds(args.seed)
    
    # --
    # IO
    
    lm_weights = torch.load(args.lm_weights_path)
    
    n_tok   = lm_weights['encoder.encoder.weight'].shape[0]
    n_class = lm_weights['decoder.layers.1.lin.weight'].shape[0]
    
    X = np.load(args.X)
    
    # Sort validation data by length, longest to shortest, for efficiency
    o = np.argsort([len(xx) for xx in X])[::-1]
    X = X[o]
    
    dataloaders = {
        "inference" : DataLoader(
            dataset=RaggedDataset(X, y=torch.zeros(len(X)).long() - 1),
            batch_size=batch_size,
            collate_fn=text_collate_fn,
            shuffle=False,
            num_workers=1,
        )
    }
    
    # --
    # Define model
    
    classifier = TextClassifier(
        bptt         = bptt,
        max_seq      = max_seq,
        n_class      = n_class,
        n_tok        = n_tok,
        emb_sz       = emb_sz,
        n_hid        = n_hid,
        n_layers     = n_layers,
        pad_token    = pad_token,
        head_layers  = [emb_sz * 3, 50, n_class],
        head_drops   = [0.0, 0.0],
        predict_only = True
    ).to('cuda')
    classifier.verbose = True
    classifier.load_state_dict(lm_weights, strict=True)
    
    preds, _ = classifier.predict(dataloaders, mode='inference')
    # return to correct order
    preds = to_numpy(preds)[np.argsort(o)]
    
    np.savetxt(args.outpath, preds, fmt='%.10f', delimiter='\t')