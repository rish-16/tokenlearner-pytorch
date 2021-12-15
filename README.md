# tokenlearner-pytorch
Unofficial PyTorch implementation of `TokenLearner` by Ryoo et al. from Google AI ([`abs`](https://arxiv.org/abs/2106.11297), [`pdf`](https://arxiv.org/pdf/2106.11297.pdf))

<img src="https://raw.githubusercontent.com/rish-16/tokenlearner-pytorch/main/tklr.png" width=650 />

## Installation
You can install TokenLearner via `pip`:

```
pip install tokenlearner-pytorch
```

## Usage
You can access the `TokenLearner` class from the `tokenlearner_pytorch` package. You can use this layer with a Vision Transformer, MLPMixer, or Video Vision Transformer as done in the paper.

```python
import torch
from tokenlearner_pytorch import TokenLearner

tklr = TokenLearner(S=8)
x = torch.rand(512, 32, 32, 3)
y = tklr(x) # [512, 8, 3]
```

You can also use `TokenLearner` and `TokenFuser` together with Multi-head Self-Attention as done in the paper:

```python
import torch
import torch.nn as nn
from tokenlearner_pytorch import TokenLearner, TokenFuser

mhsa = nn.MultiheadAttention(3, 1)
tklr = TokenLearner(S=8)
tkfr = TokenFuser(H=32, W=32, C=3, S=8)

x = torch.rand(512, 32, 32, 3) # a batch of images

y = tklr(x)
y = y.view(8, 512, 3)
y, _ = mhsa(y, y, y) # ignore attn weights
y = y.view(512, 8, 3)

out = tkfr(y, x) # [512, 32, 32, 3]
```

## TODO
- [ ] Add support for temporal dimension `T`
- [ ] Implement `TokenFuser` with `ViT`
- [ ] Implement `TokenFuser` with `ViViT`

## Contributions
If I've made any errors or you have any suggestions, feel free to raise an Issue or PR. All contributions welcome!!

## License
[MIT](https://github.com/rish-16/tokenlearner-pytorch/blob/main/LICENSE)
