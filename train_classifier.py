#!/usr/bin/env python

"""
    train_classifier.py
"""

import argparse

import os
import sys
import json
import torch
import numpy as np
from functools import partial

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from basenet.helpers import set_seeds, set_freeze
from ulmfit import TextClassifier, basenet_train

# --
# Helpers

class RaggedDataset(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y), 'len(X) != len(y)'
        self.X = [torch.LongTensor(xx) for xx in X]
        self.y = torch.LongTensor(y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.X)


def text_collate_fn(batch, pad_value=1):
    X, y = zip(*batch)
    
    max_len = max([len(xx) for xx in X])
    X = [F.pad(xx, pad=(max_len - len(xx), 0), value=pad_value) for xx in X]
    
    X = torch.stack(X, dim=-1)
    y = torch.LongTensor(y)
    return X, y


class SortishSampler(Sampler):
    def __init__(self, data_source, key, batch_size, batches_per_chunk=50):
        self.data_source       = data_source
        self.key               = key
        self.batch_size        = batch_size
        self.batches_per_chunk = batches_per_chunk
    
    def __len__(self):
        return len(self.data_source)
        
    def __iter__(self):
        
        idxs = np.random.permutation(len(self.data_source))
        
        # Group records into batches of similar size
        chunk_size = self.batch_size * self.batches_per_chunk
        chunks     = [idxs[i:i+chunk_size] for i in range(0, len(idxs), chunk_size)]
        idxs       = np.hstack([sorted(chunk, key=self.key, reverse=True) for chunk in chunks])
        
        # Make sure largest batch is in front (for memory management reasons)
        batches         = [idxs[i:i+self.batch_size] for i in range(0, len(idxs), self.batch_size)]
        batch_order     = np.argsort([self.key(b[0]) for b in batches])[::-1]
        batch_order[1:] = np.random.permutation(batch_order[1:])
        
        idxs            = np.hstack([batches[i] for i in batch_order])
        return iter(idxs)


class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        
    def __iter__(self):
        return iter(range(len(self.data_source)))
        
    def __len__(self):
        return len(self.data_source)


# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm-weights-path', type=str)
    parser.add_argument('--outpath', type=str, default='./.bak')
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
    
    args = parse_args()
    set_seeds(args.seed)
    
    os.makedirs(args.outpath, exist_ok=True)
    
    # --
    # IO
    
    lm_weights = torch.load(args.lm_weights)
    n_tok = lm_weights['encoder.encoder.weight'].shape[0]
    
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
    
    # Sort validation data by length, for efficiency
    o = np.argsort([len(x) for x in X_valid])
    X_valid, y_valid = X_valid[o], y_valid[o]
    
    dataloaders = {
        "train" : DataLoader(
            dataset=RaggedDataset(X_train, y_train),
            sampler=SortishSampler(X_train, key=lambda idx: len(X_train[idx]), bs=batch_size//2),
            batch_size=batch_size//2,
            collate_fn=text_collate_fn,
            num_workers=1,
        ),
        "valid" : DataLoader(
            dataset=RaggedDataset(X_valid, y_valid),
            sampler=SequentialSampler(X_valid),
            batch_size=batch_size,
            collate_fn=text_collate_fn,
            num_workers=1,
        )
    }
    
    # --
    # Define model
    
    def text_classifier_loss_fn(x, target, alpha=0, beta=0):
        assert isinstance(x, tuple), 'not isinstance(x, tuple)'
        assert len(x) == 3, 'len(x) != 3'
        
        l_x, raw_outputs, outputs = x
        
        # Cross entropy loss
        loss = F.cross_entropy(l_x, target)
        
        # Activation Regularization
        if alpha:
            loss = loss + sum(alpha * outputs[-1].pow(2).mean())
        
        # Temporal Activation Regularization (slowness)
        if beta: 
            h = raw_outputs[-1]
            if len(h) > 1:
                loss = loss + sum(beta * (h[1:] - h[:-1]).pow(2).mean())
        
        return loss
    
    classifier = TextClassifier(
        bptt      = bptt,
        max_seq   = 20 * 70,
        n_class   = n_class,
        n_tok     = n_tok,
        emb_sz    = emb_sz,
        n_hid     = n_hid,
        n_layers  = n_layers,
        pad_token = 1,
        layers    = [emb_sz * 3, 50, n_class],
        drops     = [dps[4], 0.1],
        dropouti  = dps[0],
        wdrop     = dps[1],
        dropoute  = dps[2],
        dropouth  = dps[3],
        loss_fn   = partial(text_classifier_loss_fn, alpha=2, beta=1),
    )
    _ = classifier.cuda()
    classifier.verbose = True
    # >>
    # !! Should save encoder weights separately in `finetune_lm.py`
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

