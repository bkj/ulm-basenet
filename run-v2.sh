#!/bin/bash

python prep.py
python featurize-v2.py
python finetune_lm-v2.py