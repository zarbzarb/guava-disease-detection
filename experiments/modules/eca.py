# -*- coding: utf-8 -*-
"""ECA 高效通道注意力（毕设 I3 改进模块）

结构：GAP → 1D 卷积跨通道交互（零降维）→ Sigmoid 门控。
输入输出形状完全相同，可无损插在任意特征层之后，不改变后续层索引的通道数。

原论文（Wang et al., CVPR 2020）的自适应核：k = |log2(C)/gamma + b/gamma|_odd，
gamma=2, b=1。本项目通道经 0.25 缩放后落在 32~256 区间，k=5 对该区间均为
合理取值，故固定 k=5（也避免动态 k 导致各 stage 配置不一致）。
"""
import torch
import torch.nn as nn


class ECABlock(nn.Module):
    """ECA 通道注意力，args 不含通道参数（从输入推断），yaml 里写 [-1, 1, ECABlock, []]"""

    def __init__(self, k_size: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=(2, 3), keepdim=True)              # GAP → (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(1, 2))      # (B, 1, C) 跨通道 1D 卷积
        y = self.sigmoid(y.transpose(1, 2).unsqueeze(-1))  # (B, C, 1, 1)
        return x * y.expand_as(x)


def register_eca():
    """把 ECABlock 注入 ultralytics 的模块命名空间（parse_model 从 tasks 模块 globals()
    里查名字），必须在加载含 ECABlock 的 yaml 之前调用；重复调用无副作用。"""
    import ultralytics.nn.tasks as _tasks
    if not hasattr(_tasks, "ECABlock"):
        _tasks.ECABlock = ECABlock
