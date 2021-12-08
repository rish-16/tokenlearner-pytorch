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
        H, W, C = x.shape
        x = x.view(C, H, W)
        
        mx = torch.max(x, 0)[0].unsqueeze(0)
        avg = torch.mean(x, 0).unsqueeze(0)
        combined = torch.cat([mx, avg], dim=0).unsqueeze(0)
        fmap = self.conv(combined)
        weight_map = torch.sigmoid(fmap).squeeze(0)
        out = (x * weight_map).mean(dim=(-2, -1))
        
        return out

class TokenLearner(nn.Module):
    def __init__(self, S) -> None:
        super().__init__()
        self.S = S
        self.tokenizers = [SpatialAttention() for _ in range(S)]
        
    def forward(self, x):
        T, H, W, C = x.shape
        
        
# class TokenFuser(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
        
#     def forward(self, x):
#         T, H, W, C = x.shape