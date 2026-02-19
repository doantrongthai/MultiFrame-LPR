"""ResTranOCR với TemporalTransformerFusion thay AttentionFusion."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import (
    ResNetFeatureExtractor,
    TemporalTransformerFusion,   # ← đổi import
    PositionalEncoding,
    STNBlock,
)


class ResTranOCR(nn.Module):
    """
    Pipeline:
    [B, F, 3, H, W]
        → STN (align từng frame)
        → ResNet34 (extract feature)
        → TemporalTransformerFusion (học quan hệ giữa F frames)
        → PositionalEncoding + TransformerEncoder (sequence modeling)
        → CTC head
    """
    def __init__(
        self,
        num_classes: int,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        use_stn: bool = True,
        # Tham số riêng cho TemporalTransformerFusion
        temporal_heads: int = 8,
        temporal_layers: int = 2,
        temporal_ff_dim: int = 1024,
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn

        # 1. STN
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 2. Backbone
        self.backbone = ResNetFeatureExtractor(pretrained=False)

        # 3. Temporal Transformer Fusion  ← thay AttentionFusion
        self.fusion = TemporalTransformerFusion(
            channels=self.cnn_channels,
            num_frames=5,
            num_heads=temporal_heads,
            num_layers=temporal_layers,
            ff_dim=temporal_ff_dim,
            dropout=dropout,
        )

        # 4. Spatial Transformer Encoder (sequence modeling theo W')
        self.pos_encoder = PositionalEncoding(d_model=self.cnn_channels, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # 5. CTC head
        self.head = nn.Linear(self.cnn_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, F, 3, H, W]
        Returns:
            log_softmax logits: [B, W', num_classes]
        """
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)           # [B*F, 3, H, W]

        # STN align
        if self.use_stn:
            theta = self.stn(x_flat)
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_flat = F.grid_sample(x_flat, grid, align_corners=False)

        # ResNet feature extraction
        features = self.backbone(x_flat)           # [B*F, 512, 1, W']

        # Temporal Transformer Fusion: học quan hệ giữa frames
        fused = self.fusion(features)              # [B, 512, 1, W']

        # Chuẩn bị cho Spatial Transformer: [B, W', 512]
        seq = fused.squeeze(2).permute(0, 2, 1)

        # Spatial positional encoding + Transformer
        seq = self.pos_encoder(seq)
        seq = self.transformer(seq)                # [B, W', 512]

        # CTC head
        out = self.head(seq)                       # [B, W', num_classes]
        return out.log_softmax(2)
