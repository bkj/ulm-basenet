#!/usr/bin/env python

"""
    finetune_lm.py
"""

import argparse

import os
import sys
import torch
import pickle
import numpy as np

from basenet.helpers import to_numpy, set_seeds, set_freeze
from ulmfit import LanguageModelLoader, LanguageModel, basenet_train

assert torch.__version__.split('.')[1] == '3', 'Downgrade to pytorch==0.3.2 (for now)'

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm-weights-path', type=str)
    parser.add_argument('--lm-itos-path', type=str)
    parser.add_argument('--itos-path', type=str)
    
    parser.add_argument('--X-train', type=str)
    parser.add_argument('--X-valid', type=str)
    parser.add_argument('--outpath', type=str)
    
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()

# --
# Helpers

def load_lm_weights(lm_weights_path, lm_itos_path, itos_path):
    lm_weights = torch.load(lm_weights_path, map_location=lambda storage, loc: storage)
    
    lm_itos = pickle.load(open(lm_itos_path, 'rb'))
    lm_stoi = {v:k for k,v in enumerate(lm_itos)}
    
    itos = pickle.load(open(itos_path, 'rb'))
    n_tok = len(itos)
    
    # Adjust vocabulary to match finetuning corpus
    lm_enc_weights = to_numpy(lm_weights['0.encoder.weight'])
    
    tmp = np.zeros((n_tok, lm_enc_weights.shape[1]), dtype=np.float32)
    tmp += lm_enc_weights.mean(axis=0)
    for i, w in enumerate(itos):
        if w in lm_stoi:
            tmp[i] = lm_enc_weights[lm_stoi[w]]
    
    lm_weights['0.encoder.weight']                    = torch.Tensor(tmp.copy())
    lm_weights['0.encoder_with_dropout.embed.weight'] = torch.Tensor(tmp.copy())
    lm_weights['1.decoder.weight']                    = torch.Tensor(tmp.copy())
    
    return lm_weights, n_tok

# --
# Run

if __name__ == "__main__":
    # --
    # Params
    
    emb_sz, nhid, nlayers = 400, 1150, 3
    bptt  = 70
    bs    = 52
    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.7
    lrs   = 1e-3
    wd    = 1e-7
    
    args = parse_args()
    set_seeds(args.seed)
    
    os.makedirs(args.outpath, exist_ok=True)
    
    # --
    # Load model
    
    print('finetune_lm.py: loading LM weights', file=sys.stderr)
    lm_weights, n_tok = load_lm_weights(args.lm_weights_path, args.lm_itos_path, args.itos_path)
    
    language_model = LanguageModel(
        n_tok     = n_tok, 
        emb_sz    = emb_sz,
        nhid      = nhid, 
        nlayers   = nlayers, 
        pad_token = 1, 
        dropouti  = drops[0],
        dropout   = drops[1],
        wdrop     = drops[2],
        dropoute  = drops[3],
        dropouth  = drops[4],
    ).to('cuda')
    print(language_model, file=sys.stderr)
    
    language_model.verbose = True
    language_model.load_weights(lm_weights)
    set_freeze(language_model, False)
    _ = language_model.train()
    
    # --
    # Load data
    
    X_train = np.load(args.X_train)
    X_valid = np.load(args.X_valid)
    
    fitist = []
    dataloaders = {
        "train" : LanguageModelLoader(np.concatenate(X_train), bs=bs, bptt=bptt),
        "valid" : LanguageModelLoader(np.concatenate(X_valid), bs=bs, bptt=bptt),
    }
    
    # Finetune decoder
    set_freeze(language_model.encoder.rnns, True)
    set_freeze(language_model.encoder.dropouths, True)
    lm_ft_last = basenet_train(
        language_model,
        dataloaders,
        num_epochs=1,
        lr_breaks=[0, 0.5, 1],
        lr_vals=[lrs / 64, lrs / 2, lrs / 64],
        adam_betas=(0.8, 0.99),
        weight_decay=wd,
        save_prefix=os.path.join(args.outpath, 'lm_ft_last'),
    )
    
    # Finetune end-to-end
    set_freeze(language_model.encoder, False)
    lm_ft_all = basenet_train(
        language_model,
        dataloaders,
        num_epochs=15,
        lr_breaks=[0, 15 / 10, 15],
        lr_vals=[lrs / 20, lrs, lrs / 20],
        adam_betas=(0.8, 0.99),
        weight_decay=wd,
        save_prefix=os.path.join(args.outpath, 'lm_ft_final'),
    )