import torch
from torch import nn
import torch.nn.functional as F
from gaussiansplatting.utils.general_utils import inverse_sigmoid

# -----------------------------
# UV Atlas Smoother (CNN / mini-UNet)
# -----------------------------
class UVResSmoother(nn.Module):
    """轻量残差CNN：卷积本身就是邻域传播 => 内置平滑。
    输出为 residual，初始近似恒等（最后一层权重置0）。
    """
    def __init__(self, c: int, hidden: int = None):
        super().__init__()
        h = hidden or c
        self.net = nn.Sequential(
            nn.Conv2d(c, h, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, h, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, c, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class UVMiniUNetSmoother(nn.Module):
    """更像 U-Net 的结构（1次下采样+上采样+skip），仍然做 residual，初始近似恒等。"""
    def __init__(self, c: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.down = nn.AvgPool2d(2)
        self.mid = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec = nn.Sequential(
            nn.Conv2d(2 * c, c, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
        )
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x):
        skip = self.enc(x)
        y = self.down(skip)
        y = self.mid(y)
        y = self.up(y)
        y = torch.cat([y, skip], dim=1)
        y = self.dec(y)
        return x + y


class UVAtlasHeads(nn.Module):
    """UV特征 -> (rgb, opacity_logit)"""
    def __init__(self, feat_dim: int, opacity_init: float = 0.1):
        super().__init__()
        self.rgb_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 3),
        )
        self.op_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
        )
        # 初始化：rgb 输出=0 => sigmoid=0.5；opacity 输出为较小值（避免一开始过多半透明点）
        nn.init.zeros_(self.rgb_head[-1].weight)
        nn.init.zeros_(self.rgb_head[-1].bias)
        nn.init.zeros_(self.op_head[-1].weight)
        self.op_head[-1].bias.data.fill_(float(inverse_sigmoid(torch.tensor(opacity_init))))


    def forward(self, feat):
        rgb = torch.sigmoid(self.rgb_head(feat))         # (N,3) in [0,1]
        op  = torch.sigmoid(self.op_head(feat))          # (N,1) in [0,1]

        return rgb, op
