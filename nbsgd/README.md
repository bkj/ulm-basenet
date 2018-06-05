### ulm-basenet/nbsgd

ULMFit is very high accuracy -- but inference is slow, even on a GPU.  We'd like a model that is high accuracy and has fast inference.

One approach to this is [knowledge distillation](https://arxiv.org/abs/1503.02531), where we train a smaller, faster model to predict the outputs of a larger, slower model.

Specifically, in normal training, we minimize
```
small_model_logits = small_model(X)

loss = F.cross_entropy(small_model_logits, y)
```

but in knowledge distillation we minimize something like

```
T = 1                # Optional temperature parameter (higher means target logits are "softer")
distill_weight = 1.0 # Relative weight of distillation loss

# Model predictions
small_model_logits = small_model(X)
large_model_logits = large_model(X)

ce_loss      = F.cross_entropy(small_model_logits, y)
distill_loss = compute_distill_loss(small_model_logits, large_model_logits)

loss = ce_loss + hot_weight * distill_loss
```

I don't think the exact definition of `compute_distill_loss` is super important (different papers do different things).  In this code we'll use:
```
def compute_distill_loss(small_model_logits, large_model_logits, T=1):
    """ T is optional temperature parameter -- higher values make the targets 'softer' """
    distill_loss = (F.softmax(large_model_logits / T, dim=-1) * F.log_softmax(small_model_logits, dim=-1))
    distill_loss = - distill_loss.sum(dim=-1).mean()
    return distill_loss
```

Note that we can modify this slightly to be a semi-supervised loss function by ignoring observations w/o labels in the `ce_loss` computation.  In this IMDB demo, using both the `train` and `unlabeled` folds for training `nbsgd` is definitely beneficial.

#### Results

Without much hyperparameter tuning, we get:

```
...
{"epoch": 29, "lr": 1.481481481481713e-07, "test_acc": 0.9462, "time": 52.78358459472656}
```

So:
- Training this `nbsgd` model w/ distillation yields accuracy around 0.946
- Training this `nbsgd` model from scratch yields accuracy around 0.921
- The original [ULMFit](https://arxiv.org/abs/1801.06146) model has accuracy around 0.948
- According to the [ULMFit](https://arxiv.org/abs/1801.06146) paper, previous state of the art was 0.941

Thus, our simple `nbsgd` model trained w/ distillation gets pretty close to ULMFit accuracy, still beating previous SOTA. 

However, the `nbsgd` model is significantly faster:
```
    Model   Chip    Speedup over ULMFit
    
    ULMFit  GPU     1x
    nbsgd   CPU     ~150x
    nbsgd   GPU     ~400x
```

#### Notes

- If you look closely, you'll see that `nbsgd` and `ulm-basenet` don't use the same text preprocessing or tokenization -- a nice property of knowledge distillation is that it doesn't matter.
- My ULMFit accuracy is lower than the 0.954 reported in the paper because they use a bidirectional LM -- if I added that to `ulm-basenet`, I'd expect an accuracy bump.
