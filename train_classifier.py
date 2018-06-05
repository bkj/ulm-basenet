#!/usr/bin/env python

"""
    train_classifier.py
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
from torch.utils.data.sampler import SequentialSampler

from basenet.helpers import set_seeds, set_freeze
from basenet.text.data import RaggedDataset, SortishSampler, text_collate_fn

from ulmfit import TextClassifier, basenet_train

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm-weights-path', type=str)
    parser.add_argument('--outpath', type=str)
    parser.add_argument('--X-train', type=str)
    parser.add_argument('--y-train', type=str)
    parser.add_argument('--X-valid', type=str)
    parser.add_argument('--y-valid', type=str)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

# --
# Run

if __name__ == "__main__":
    # --
    # Params
    
    bptt, emb_sz, n_hid, n_layers, batch_size = 70, 400, 1150, 3, 48
    dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])
    lr  = 3e-3
    lrm = 2.6
    lrs = np.array([lr / (lrm ** i) for i in range(5)[::-1]])
    max_seq = 20 * 70
    pad_token = 1
    
    args = parse_args()
    set_seeds(args.seed)
    
    os.makedirs(args.outpath, exist_ok=True)
    
    # --
    # IO
    
    X_train = np.load(args.X_train)
    y_train = np.load(args.y_train).squeeze()
    
    X_valid = np.load(args.X_valid)
    y_valid = np.load(args.y_valid).squeeze()
    
    # Map labels to sequential ints
    ulabs = np.unique(y_train)
    n_class = len(ulabs)
    
    lab_lookup = dict(zip(ulabs, range(len(ulabs))))
    y_train = np.array([lab_lookup[l] for l in y_train])
    y_valid = np.array([lab_lookup[l] for l in y_valid])
    json.dump(
        {str(k):v for k,v in lab_lookup.items()}, 
        open(os.path.join(args.outpath, 'classes.json'), 'w')
    )
    
    # Sort validation data by length, longest to shortest, for efficiency
    o = np.argsort([len(x) for x in X_valid])[::-1]
    X_valid, y_valid = X_valid[o], y_valid[o]
    
    dataloaders = {
        "train" : DataLoader(
            dataset=RaggedDataset(X_train, y_train),
            sampler=SortishSampler(X_train, batch_size=batch_size//2),
            batch_size=batch_size//2,
            collate_fn=text_collate_fn,
            num_workers=1,
            pin_memory=True,
        ),
        "valid" : DataLoader(
            dataset=RaggedDataset(X_valid, y_valid),
            sampler=SequentialSampler(X_valid),
            batch_size=batch_size,
            collate_fn=text_collate_fn,
            num_workers=1,
            pin_memory=True,
        )
    }
    
    # --
    # Define model
    
    def text_classifier_loss_fn(x, target, alpha=0, beta=0):
        assert isinstance(x, tuple), 'not isinstance(x, tuple)'
        assert len(x) == 3, 'len(x) != 3'
        
        l_x, last_raw_output, last_output = x
        
        # Cross entropy loss
        loss = F.cross_entropy(l_x, target)
        
        # Activation Regularization
        if alpha:
            loss = loss + sum(alpha * last_output.pow(2).mean())
        
        # Temporal Activation Regularization (slowness)
        if beta: 
            if len(last_raw_output) > 1:
                loss = loss + sum(beta * (last_raw_output[1:] - last_raw_output[:-1]).pow(2).mean())
        
        return loss
        
    lm_weights = torch.load(args.lm_weights_path)
    n_tok = lm_weights['encoder.encoder.weight'].shape[0]
    
    classifier = TextClassifier(
        bptt        = bptt,
        max_seq     = max_seq,
        n_class     = n_class,
        n_tok       = n_tok,
        emb_sz      = emb_sz,
        n_hid       = n_hid,
        n_layers    = n_layers,
        pad_token   = pad_token,
        head_layers = [emb_sz * 3, 50, n_class],
        head_drops  = [dps[4], 0.1],
        dropouti    = dps[0],
        wdrop       = dps[1],
        dropoute    = dps[2],
        dropouth    = dps[3],
        loss_fn     = partial(text_classifier_loss_fn, alpha=2, beta=1),
    ).to('cuda')
    classifier.verbose = True
    print(classifier, file=sys.stderr)
    
    # >>
    # !! Should maybe save encoder weights separately in `finetune_lm.py`
    weights_to_drop = [k for k in lm_weights.keys() if 'decoder.' in k]
    for k in weights_to_drop:
        del lm_weights[k]
    # <<
    
    classifier.load_state_dict(lm_weights, strict=False)
    set_freeze(classifier, False)
    
    # --
    # Train
    
    # Finetune decoder
    set_freeze(classifier.encoder.encoder, True)
    set_freeze(classifier.encoder.dropouti, True)
    set_freeze(classifier.encoder.rnns, True)
    set_freeze(classifier.encoder.dropouths, True)
    
    class_ft_dec = basenet_train(
        classifier,
        dataloaders,
        num_epochs=1,
        lr_breaks=[0, 1/3, 1],
        lr_vals=[lrs / 8, lrs, lrs / 8],
        adam_betas=(0.7, 0.99),
        weight_decay=0,
        clip_grad_norm=25,
        save_prefix=os.path.join(args.outpath, 'cl_ft_last1'),
    )
    
    # Finetune last layer
    set_freeze(classifier.encoder.rnns[-1], False)
    set_freeze(classifier.encoder.dropouths[-1], False)
    class_ft_last = basenet_train(
        classifier,
        dataloaders,
        num_epochs=1,
        lr_breaks=[0, 1/3, 1],
        lr_vals=[lrs / 8, lrs, lrs / 8],
        adam_betas=(0.7, 0.99),
        weight_decay=0,
        clip_grad_norm=25,
        save_prefix=os.path.join(args.outpath, 'cl_ft_last2'),
    )
    
    # Finetune end-to-end
    set_freeze(classifier, False)
    class_ft_all = basenet_train(
        classifier,
        dataloaders,
        num_epochs=14,
        lr_breaks=[0, 14/10, 14],
        lr_vals=[lrs / 32, lrs, lrs / 32],
        adam_betas=(0.7, 0.99),
        weight_decay=0,
        clip_grad_norm=25,
        save_prefix=os.path.join(args.outpath, 'cl_final'),
    )

