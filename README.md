# tokenlearner-pytorch
Unofficial PyTorch implementation of `TokenLearner` by Ryoo et al. from Google AI ([`abs`](https://arxiv.org/abs/2106.11297), [`pdf`](https://arxiv.org/pdf/2106.11297.pdf))

## Installation
You can install TokenLearner via `pip`:

```
pip install tokenlearner-pytorch
```

## Usage
You can access the `TokenLearner` class from the `tokenlearner_pytorch` package:

```python
import torch
from tokenlearner_pytorch import TokenLearner

tklr = TokenLearner(S=8)
x = torch.rand(512, 32, 32, 3)
y = tklr(x) # [512, 8, 3]
```

## TODO
- [ ] Add support for temporal dimension `T`
- [ ] Implement `TokenFuser` with `ViT`
- [ ] Implement `TokenFuser` with `ViViT`

## Contributions
If I've made any errors or you have any suggestions, feel free to raise an Issue or PR. All contributions welcome!!

## License
[MIT](https://github.com/rish-16/tokenlearner-pytorch/blob/main/LICENSE)
