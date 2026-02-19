"""Config dataclass cho training pipeline."""
from dataclasses import dataclass, field
from typing import Dict, List
import torch


@dataclass
class Config:

    # ── Experiment ───────────────────────────────────────────────────────────
    EXPERIMENT_NAME:    str  = "restran_sr"
    AUGMENTATION_LEVEL: str  = "full"
    MODEL_TYPE:         str  = "restran"
    USE_STN:            bool = True

    # ── Data ─────────────────────────────────────────────────────────────────
    DATA_ROOT:      str = "data/train"
    TEST_DATA_ROOT: str = "data/public_test"
    VAL_SPLIT_FILE: str = "data/val_tracks.json"
    IMG_HEIGHT:     int = 32
    IMG_WIDTH:      int = 128

    # ── Chars ────────────────────────────────────────────────────────────────
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # ── Training ─────────────────────────────────────────────────────────────
    BATCH_SIZE:          int   = 16
    LEARNING_RATE:       float = 5e-4
    EPOCHS:              int   = 30
    SEED:                int   = 42
    NUM_WORKERS:         int   = 4
    WEIGHT_DECAY:        float = 1e-4
    GRAD_CLIP:           float = 5.0
    SPLIT_RATIO:         float = 0.9
    USE_CUDNN_BENCHMARK: bool  = False

    # ── CRNN ─────────────────────────────────────────────────────────────────
    HIDDEN_SIZE: int   = 256
    RNN_DROPOUT: float = 0.25

    # ── ResTranOCR Spatial Transformer ───────────────────────────────────────
    TRANSFORMER_HEADS:   int   = 8
    TRANSFORMER_LAYERS:  int   = 3
    TRANSFORMER_FF_DIM:  int   = 2048
    TRANSFORMER_DROPOUT: float = 0.1

    # ── Temporal Transformer Fusion ──────────────────────────────────────────
    TEMPORAL_HEADS:  int = 8
    TEMPORAL_LAYERS: int = 2
    TEMPORAL_FF_DIM: int = 1024

    # ── SR Module ────────────────────────────────────────────────────────────
    USE_SR:           bool  = False
    SR_SCALE:         int   = 4
    SR_NUM_BLOCK:     int   = 23
    SR_PRETRAIN_PATH: str   = "weights/RealESRGAN_x4plus.pth"
    SR_LOSS_WEIGHT:   float = 0.1
    SR_LR_FACTOR:     int   = 10
    SR_FREEZE:        bool  = True
    SR_UNFREEZE_EPOCH: int  = 10

    # ── Cách dùng SR:
    #
    # [Giai đoạn 1] Frozen SR — chỉ train OCR:
    #   USE_SR            = True
    #   SR_PRETRAIN_PATH  = "weights/RealESRGAN_x4plus.pth"
    #   SR_FREEZE         = True
    #   SR_UNFREEZE_EPOCH = 999    ← không bao giờ unfreeze
    #
    # [Giai đoạn 2] End-to-end — unfreeze SR sau N epoch:
    #   USE_SR            = True
    #   SR_PRETRAIN_PATH  = "weights/RealESRGAN_x4plus.pth"
    #   SR_FREEZE         = True
    #   SR_UNFREEZE_EPOCH = 10     ← unfreeze từ epoch 10
    #
    # [Baseline] Không SR:
    #   USE_SR = False

    # ── Device & Output ──────────────────────────────────────────────────────
    DEVICE: torch.device = field(
        default_factory=lambda: torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    )
    OUTPUT_DIR: str = "results"

    # ── Derived (auto-computed) ───────────────────────────────────────────────
    CHAR2IDX:         Dict[str, int] = field(default_factory=dict, init=False)
    IDX2CHAR:         Dict[int, str] = field(default_factory=dict, init=False)
    NUM_CLASSES:      int            = field(default=0,   init=False)
    FREEZE_SR_EPOCHS: int            = field(default=0,   init=False)  # alias cho trainer

    def __post_init__(self):
        self.CHAR2IDX         = {c: i + 1 for i, c in enumerate(self.CHARS)}
        self.IDX2CHAR         = {i + 1: c for i, c in enumerate(self.CHARS)}
        self.NUM_CLASSES      = len(self.CHARS) + 1
        self.FREEZE_SR_EPOCHS = self.SR_UNFREEZE_EPOCH  # trainer đọc key này