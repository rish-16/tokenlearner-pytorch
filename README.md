# tokenlearner-pytorch
Unofficial PyTorch implementation of `TokenLearner` by Ryoo et al. from Google AI [[`abs`](https://arxiv.org/abs/2106.11297), [`pdf`](https://arxiv.org/pdf/2106.11297.pdf)]

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

x = torch.rand(3, 32, 32)
y = tklr(x)
```

## TODO
- [ ] Add batch support to Spatial Attention
- [ ] Implement TokenLearner
- [ ] Implement TokenFuser

## Contributions
If I've made any errors or you have any suggestions, feel free to raise an Issue or PR. All contributions welcome!!

## License
[MIT](https://github.com/rish-16/tokenlearner-pytorch/blob/main/LICENSE)