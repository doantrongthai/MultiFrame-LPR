"""ResTranOCR: ResNet34 + Transformer architecture (Advanced) with STN + SR branch."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import (
    ResNetFeatureExtractor,
    TemporalTransformerFusion,
    PositionalEncoding,
    STNBlock,
)
from src.models.rrdbnet import RRDBNet, load_pretrained_rrdbnet


class ResTranOCR(nn.Module):
    """
    Modern OCR architecture: [Optional SR] → [Optional STN] → ResNet34
                              → TemporalTransformerFusion → Transformer → CTC Head

    SR Branch (Real-ESRGAN RRDBNet):
        - Takes LR frames, upsamples x4 to produce SR frames
        - SR frames are resized back to OCR input size before backbone
        - SR loss (L1 vs HR) computed externally in trainer using sr_output
        - Pretrained weights from RealESRGAN_x4plus.pth can be loaded

    Args:
        num_classes: Number of output classes (charset size + blank).
        transformer_heads: Attention heads in spatial transformer.
        transformer_layers: Layers in spatial transformer.
        transformer_ff_dim: Feed-forward dim in spatial transformer.
        dropout: Dropout rate.
        use_stn: Whether to use Spatial Transformer Network.
        use_sr: Whether to use SR branch (RRDBNet) before backbone.
        sr_scale: SR upscale factor (4 matches RealESRGAN_x4plus.pth).
        sr_num_block: Number of RRDB blocks (23 matches pretrain).
        sr_pretrain_path: Path to pretrained .pth file. None = random init.
        freeze_sr_epochs: Freeze SR branch for first N epochs (warm up OCR first).
    """

    def __init__(
        self,
        num_classes: int,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        use_stn: bool = True,
        use_sr: bool = False,
        sr_scale: int = 4,
        sr_num_block: int = 23,
        sr_pretrain_path: str = None,
        freeze_sr_epochs: int = 0,
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        self.use_sr = use_sr
        self.freeze_sr_epochs = freeze_sr_epochs
        self._current_epoch = 0

        # 1. SR Branch (Real-ESRGAN RRDBNet)
        if self.use_sr:
            self.sr_net = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                scale=sr_scale,
                num_feat=64,
                num_block=sr_num_block,
                num_grow_ch=32,
            )
            if sr_pretrain_path is not None:
                import torch
                load_pretrained_rrdbnet(
                    self.sr_net,
                    sr_pretrain_path,
                    device=torch.device('cpu'),  # Moved to device later by trainer
                    strict=True,
                )
            if freeze_sr_epochs > 0:
                self._freeze_sr()
                print(f"❄️  SR branch frozen for first {freeze_sr_epochs} epochs.")

        # 2. Spatial Transformer Network (STN)
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 3. Backbone: ResNet34 (custom stride for OCR)
        self.backbone = ResNetFeatureExtractor(pretrained=False)

        # 4. Temporal Transformer Fusion (replaces AttentionFusion)
        self.fusion = TemporalTransformerFusion(channels=self.cnn_channels)

        # 5. Spatial Positional Encoding + Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model=self.cnn_channels, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # 6. Prediction Head
        self.head = nn.Linear(self.cnn_channels, num_classes)

    # ------------------------------------------------------------------
    # SR freeze / unfreeze helpers (called by trainer)
    # ------------------------------------------------------------------

    def _freeze_sr(self) -> None:
        """Freeze all SR branch parameters."""
        if self.use_sr:
            for p in self.sr_net.parameters():
                p.requires_grad = False

    def _unfreeze_sr(self) -> None:
        """Unfreeze all SR branch parameters."""
        if self.use_sr:
            for p in self.sr_net.parameters():
                p.requires_grad = True

    def set_epoch(self, epoch: int) -> None:
        """Called by trainer at start of each epoch to manage SR freeze."""
        self._current_epoch = epoch
        if self.use_sr and self.freeze_sr_epochs > 0:
            if epoch == self.freeze_sr_epochs:
                self._unfreeze_sr()
                print(f"🔓 SR branch unfrozen at epoch {epoch + 1}.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_sr: bool = False,
    ) -> tuple:
        """
        Args:
            x: LR input frames [Batch, Frames, 3, H, W]
            return_sr: If True, also return SR output for loss computation.

        Returns:
            If return_sr=False: logits [Batch, Seq_Len, Num_Classes]
            If return_sr=True:  (logits, sr_output)
                sr_output: [Batch, Frames, 3, H*scale, W*scale] or None
        """
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)  # [B*F, C, H, W]

        # --- SR Branch ---
        sr_output = None
        if self.use_sr:
            # Normalize from [-1,1] (OCR norm) → [0,1] for SR network
            x_for_sr = (x_flat * 0.5 + 0.5).clamp(0, 1)

            sr_flat = self.sr_net(x_for_sr)           # [B*F, 3, H*scale, W*scale]
            sr_flat = sr_flat.clamp(0, 1)

            if return_sr:
                sr_output = sr_flat.view(b, f, c, sr_flat.shape[2], sr_flat.shape[3])

            # Resize SR output back to OCR input size, renormalize to [-1,1]
            x_flat = F.interpolate(sr_flat, size=(h, w), mode='bilinear', align_corners=False)
            x_flat = (x_flat - 0.5) / 0.5  # Back to [-1, 1]

        # --- STN ---
        if self.use_stn:
            theta = self.stn(x_flat)
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_flat = F.grid_sample(x_flat, grid, align_corners=False)

        # --- Backbone ---
        features = self.backbone(x_flat)   # [B*F, 512, 1, W']

        # --- Temporal Transformer Fusion ---
        fused = self.fusion(features)      # [B, 512, 1, W'] (interface preserved)

        # --- Spatial Transformer ---
        seq_input = fused.squeeze(2).permute(0, 2, 1)  # [B, W', C]
        seq_input = self.pos_encoder(seq_input)
        seq_out = self.transformer(seq_input)            # [B, W', C]

        # --- CTC Head ---
        out = self.head(seq_out)                         # [B, W', Num_Classes]
        logits = out.log_softmax(2)

        if return_sr:
            return logits, sr_output
        return logits