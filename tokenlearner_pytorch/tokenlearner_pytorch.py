import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1,1), stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        
        self.sgap = nn.AvgPool2d(2)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, C, H, W)
        
        mx = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        combined = torch.cat([mx, avg], dim=1)
        fmap = self.conv(combined)
        weight_map = torch.sigmoid(fmap)
        out = (x * weight_map).mean(dim=(-2, -1))
        
        return out

class TokenLearner(nn.Module):
    def __init__(self, S) -> None:
        super().__init__()
        self.S = S
        self.tokenizers = [SpatialAttention() for _ in range(S)]
        
    def forward(self, x):
        B, _, _, C = x.shape
        Z = torch.Tensor(B, self.S, C)
        for i in range(self.S):
            Ai = self.tokenizers[i](x) # [B, C]
            Z[:, i, :] = Ai
        return Z
    
class TokenFuser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        pass