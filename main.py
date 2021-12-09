import torch
import torch.nn as nn
from tokenlearner_pytorch import TokenLearner, TokenFuser

mhsa = nn.MultiheadAttention(3, 1)
tklr = TokenLearner(S=8)
tkfr = TokenFuser(32, 32, 3, S=8)

x = torch.rand(512, 32, 32, 3)

y = tklr(x)
y = y.view(8, 512, 3)
y, _ = mhsa(y, y, y) # ignore attn weights
y = y.view(512, 8, 3)

out = tkfr(y, x)
print (out.shape)