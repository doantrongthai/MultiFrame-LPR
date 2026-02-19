#!/usr/bin/env python3
"""Main entry point for OCR training pipeline."""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR
from src.training.trainer import Trainer
from src.utils.common import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Multi-Frame OCR for License Plate Recognition"
    )
    parser.add_argument("-n", "--experiment-name", type=str, default=None)
    parser.add_argument("-m", "--model", type=str, choices=["crnn", "restran"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", "--learning-rate", type=float, default=None, dest="learning_rate")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--transformer-heads", type=int, default=None)
    parser.add_argument("--transformer-layers", type=int, default=None)
    parser.add_argument("--aug-level", type=str, choices=["full", "light"], default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--no-stn", action="store_true")
    parser.add_argument("--submission-mode", action="store_true")

    # SR arguments
    parser.add_argument(
        "--use-sr", action="store_true",
        help="Enable Real-ESRGAN SR branch (end-to-end SR + OCR)",
    )
    parser.add_argument(
        "--sr-pretrain", type=str, default=None,
        help="Path to RealESRGAN_x4plus.pth pretrained weights",
    )
    parser.add_argument(
        "--sr-loss-weight", type=float, default=0.1,
        help="Weight for SR L1 loss (default: 0.1)",
    )
    parser.add_argument(
        "--sr-lr-factor", type=int, default=10,
        help="LR divisor for SR branch vs OCR (default: 10)",
    )
    parser.add_argument(
        "--freeze-sr-epochs", type=int, default=5,
        help="Freeze SR branch for first N epochs to warm up OCR (default: 5)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    arg_to_config = {
        'experiment_name': 'EXPERIMENT_NAME',
        'model': 'MODEL_TYPE',
        'epochs': 'EPOCHS',
        'batch_size': 'BATCH_SIZE',
        'learning_rate': 'LEARNING_RATE',
        'data_root': 'DATA_ROOT',
        'seed': 'SEED',
        'num_workers': 'NUM_WORKERS',
        'hidden_size': 'HIDDEN_SIZE',
        'transformer_heads': 'TRANSFORMER_HEADS',
        'transformer_layers': 'TRANSFORMER_LAYERS',
    }
    for arg_name, config_name in arg_to_config.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, value)

    if args.aug_level is not None:
        config.AUGMENTATION_LEVEL = args.aug_level
    if args.no_stn:
        config.USE_STN = False

    # SR config
    config.USE_SR           = args.use_sr
    config.SR_PRETRAIN_PATH = args.sr_pretrain
    config.SR_LOSS_WEIGHT   = args.sr_loss_weight
    config.SR_LR_FACTOR     = args.sr_lr_factor
    config.FREEZE_SR_EPOCHS = args.freeze_sr_epochs

    config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    seed_everything(config.SEED)

    print(f"🚀 Configuration:")
    print(f"   EXPERIMENT: {config.EXPERIMENT_NAME}")
    print(f"   MODEL: {config.MODEL_TYPE}")
    print(f"   USE_STN: {config.USE_STN}")
    print(f"   USE_SR: {config.USE_SR}")
    if config.USE_SR:
        print(f"   SR_PRETRAIN: {config.SR_PRETRAIN_PATH}")
        print(f"   SR_LOSS_WEIGHT: {config.SR_LOSS_WEIGHT}")
        print(f"   FREEZE_SR_EPOCHS: {config.FREEZE_SR_EPOCHS}")
    print(f"   DATA_ROOT: {config.DATA_ROOT}")
    print(f"   EPOCHS: {config.EPOCHS}")
    print(f"   BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"   LEARNING_RATE: {config.LEARNING_RATE}")
    print(f"   DEVICE: {config.DEVICE}")

    if not os.path.exists(config.DATA_ROOT):
        print(f"❌ ERROR: Data root not found: {config.DATA_ROOT}")
        sys.exit(1)

    # HR image size = LR * SR scale (4)
    sr_scale = 4
    hr_img_height = config.IMG_HEIGHT * sr_scale
    hr_img_width  = config.IMG_WIDTH  * sr_scale

    common_ds_params = {
        'split_ratio':       config.SPLIT_RATIO,
        'img_height':        config.IMG_HEIGHT,
        'img_width':         config.IMG_WIDTH,
        'char2idx':          config.CHAR2IDX,
        'val_split_file':    config.VAL_SPLIT_FILE,
        'seed':              config.SEED,
        'augmentation_level': config.AUGMENTATION_LEVEL,
        'use_sr':            config.USE_SR,
        'hr_img_height':     hr_img_height,
        'hr_img_width':      hr_img_width,
    }

    if args.submission_mode:
        print("\n📌 SUBMISSION MODE ENABLED")
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT, mode='train', full_train=True, **common_ds_params,
        )
        test_loader = None
        if os.path.exists(config.TEST_DATA_ROOT):
            test_ds = MultiFrameDataset(
                root_dir=config.TEST_DATA_ROOT,
                mode='val',
                img_height=config.IMG_HEIGHT,
                img_width=config.IMG_WIDTH,
                char2idx=config.CHAR2IDX,
                seed=config.SEED,
                is_test=True,
                use_sr=False,
            )
            test_loader = DataLoader(
                test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS, pin_memory=True,
            )
        val_loader = None
    else:
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT, mode='train', **common_ds_params,
        )
        val_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT, mode='val', **common_ds_params,
        )
        val_loader = None
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS, pin_memory=True,
            )
        test_loader = None

    if len(train_ds) == 0:
        print("❌ Training dataset is empty!")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # Build model
    if config.MODEL_TYPE == "restran":
        model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            use_sr=config.USE_SR,
            sr_scale=sr_scale,
            sr_num_block=23,
            sr_pretrain_path=config.SR_PRETRAIN_PATH,
            freeze_sr_epochs=config.FREEZE_SR_EPOCHS if config.USE_SR else 0,
        ).to(config.DEVICE)
    else:
        model = MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            rnn_dropout=config.RNN_DROPOUT,
            use_stn=config.USE_STN,
        ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model ({config.MODEL_TYPE}): {total_params:,} total, {trainable_params:,} trainable")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR,
    )

    trainer.fit()

    if args.submission_mode and test_loader is not None:
        print("\n" + "="*60)
        print("📝 GENERATING SUBMISSION FILE")
        print("="*60)

        exp_name = config.EXPERIMENT_NAME
        best_model_path = os.path.join(config.OUTPUT_DIR, f"{exp_name}_best.pth")
        if os.path.exists(best_model_path):
            print(f"📦 Loading best checkpoint: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))

        trainer.predict_test(test_loader, output_filename=f"submission_{exp_name}_final.txt")


if __name__ == "__main__":
    main()