### basenet-ulmfit

Implementation of [ulmfit](https://github.com/fastai/fastai/tree/master/courses/dl2/imdb_scripts) using [basenet](https://github.com/bkj/basenet) wrappers.

Code in `ulmfit.py` is adapted directly from [fastai](https://github.com/fastai/fastai) code.

#### Installation

```
conda create -n ulm_env python=3.6 pip -y
source activate ulm_env

# pytorch
conda install -c pytorch pytorch==0.4.1 -y

# spacy (for tokenization)
conda install -c conda-forge spacy==2.0.18 -y
python -m spacy download en

# additional requirements
pip install -r requirements.txt
```

#### Usage

See `./run.sh` for usage.

#### Debugging Notes

If you get an error like:
```
ValueError: 1792000 exceeds max_bin_len(1048576)
```
you can try [downgrading msgpack](https://github.com/explosion/spaCy/issues/2994):
```
msgpack==0.5.6
```