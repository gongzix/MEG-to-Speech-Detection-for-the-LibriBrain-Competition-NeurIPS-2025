import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        x_norm = self.layernorm(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        return attn_output


class ConvolutionModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, dim * 2, kernel_size=1)  # for GLU
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                        padding=kernel_size // 2, groups=dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, T, D]
        x = self.layernorm(x)
        x = x.transpose(1, 2)             # [B, D, T]
        x = self.pointwise_conv1(x)       # [B, 2D, T]
        x = F.glu(x, dim=1)               # [B, D, T]
        x = self.depthwise_conv(x)        # [B, D, T]
        x = F.silu(x)
        x = self.pointwise_conv2(x)       # [B, D, T]
        x = x.transpose(1, 2)             # [B, T, D]
        return self.dropout(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim, ff_expansion=4, conv_expansion=2, num_heads=8, dropout=0.1, conv_kernel_size=31):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, expansion_factor=ff_expansion, dropout=dropout)
        self.mha = MultiHeadSelfAttention(dim, num_heads=num_heads, dropout=dropout)
        self.conv = ConvolutionModule(dim, kernel_size=conv_kernel_size, dropout=dropout)
        self.ff2 = FeedForwardModule(dim, expansion_factor=ff_expansion, dropout=dropout)
        self.final_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, T, D]
        x = x + 0.5 * self.dropout(self.ff1(x))
        x = x + self.dropout(self.mha(x))
        x = x + self.dropout(self.conv(x))
        x = x + 0.5 * self.dropout(self.ff2(x))
        return self.final_norm(x)

class MEGConformerVAD(nn.Module):
    def __init__(self, in_channels=306, model_dim=128, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, model_dim)

        self.encoder = nn.ModuleList([
            ConformerBlock(
                dim=model_dim,
                ff_expansion=4,
                num_heads=num_heads,
                dropout=dropout,
                conv_kernel_size=31
            )
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 1),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: [B, C, T]
        x = x.permute(0, 2, 1)          # [B, T, C]
        x = self.input_proj(x)          # [B, T, D]
        for block in self.encoder:
            x = block(x)                # [B, T, D]
        x = self.classifier(x).squeeze(-1)  # [B, T]
        return x