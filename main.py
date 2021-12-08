import torch
from tokenlearner_pytorch import TokenLearner

tklr = TokenLearner(S=8)
x = torch.rand(512, 32, 32, 3)
y = tklr(x)
print (y.shape)