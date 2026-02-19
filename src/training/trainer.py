"""Trainer class encapsulating the training and validation loop."""
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


class Trainer:
    """Encapsulates training, validation, and inference logic.

    Supports end-to-end SR + OCR training:
        - SR loss: L1(sr_output, hr_target) weighted by config.SR_LOSS_WEIGHT
        - CTC loss: standard sequence recognition loss
        - total_loss = ctc_loss + sr_weight * sr_loss
        - Discriminative LR: SR backbone gets lower lr than OCR head
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str]
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE
        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)

        # Detect if model has SR branch
        self.use_sr = getattr(model, 'use_sr', False)
        self.sr_loss_weight = getattr(config, 'SR_LOSS_WEIGHT', 0.1)

        # Loss
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')

        # Optimizer: discriminative LR if SR branch exists
        self.optimizer = self._build_optimizer()

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=config.EPOCHS,
        )
        self.scaler = GradScaler()

        self.best_acc = 0.0
        self.current_epoch = 0

    def _build_optimizer(self) -> optim.Optimizer:
        """Build AdamW optimizer with discriminative learning rates.

        SR branch: lr / SR_LR_FACTOR (lower, already pretrained)
        OCR components: base lr
        """
        lr = self.config.LEARNING_RATE
        wd = self.config.WEIGHT_DECAY
        sr_lr_factor = getattr(self.config, 'SR_LR_FACTOR', 10)

        if self.use_sr and hasattr(self.model, 'sr_net'):
            print(f"⚙️  Discriminative LR: SR branch lr={lr/sr_lr_factor:.2e}, OCR lr={lr:.2e}")
            param_groups = [
                # SR backbone: lower lr (pretrained weights, don't destroy)
                {
                    'params': self.model.sr_net.parameters(),
                    'lr': lr / sr_lr_factor,
                    'name': 'sr_net',
                },
                # OCR components: base lr
                {
                    'params': self.model.backbone.parameters(),
                    'lr': lr,
                    'name': 'backbone',
                },
                {
                    'params': self.model.fusion.parameters(),
                    'lr': lr,
                    'name': 'fusion',
                },
                {
                    'params': self.model.transformer.parameters(),
                    'lr': lr,
                    'name': 'transformer',
                },
                {
                    'params': self.model.pos_encoder.parameters(),
                    'lr': lr,
                    'name': 'pos_encoder',
                },
                {
                    'params': self.model.head.parameters(),
                    'lr': lr,
                    'name': 'head',
                },
            ]
            # Add STN if present
            if hasattr(self.model, 'stn'):
                param_groups.append({
                    'params': self.model.stn.parameters(),
                    'lr': lr,
                    'name': 'stn',
                })
            return optim.AdamW(param_groups, lr=lr, weight_decay=wd)
        else:
            # Standard: all parameters same lr
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd,
            )

    def _get_output_path(self, filename: str) -> str:
        output_dir = getattr(self.config, 'OUTPUT_DIR', 'results')
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    def _get_exp_name(self) -> str:
        return getattr(self.config, 'EXPERIMENT_NAME', 'baseline')

    def train_one_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dict with 'loss', 'ctc_loss', 'sr_loss' (sr_loss=0 if not use_sr).
        """
        self.model.train()

        # Notify model of current epoch (for SR freeze/unfreeze)
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(self.current_epoch)

        epoch_loss = 0.0
        epoch_ctc_loss = 0.0
        epoch_sr_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}")

        for batch in pbar:
            # Unpack batch — dataset returns 6 elements now
            images, targets, target_lengths, _, _, hr_images = batch

            images = images.to(self.device)
            targets = targets.to(self.device)

            # Check if HR images are available for SR loss
            has_hr = self.use_sr and hr_images.numel() > 0
            if has_hr:
                hr_images = hr_images.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                # Forward — request SR output if we need SR loss
                if self.use_sr and has_hr:
                    preds, sr_output = self.model(images, return_sr=True)
                else:
                    preds = self.model(images, return_sr=False)
                    sr_output = None

                # CTC Loss
                preds_permuted = preds.permute(1, 0, 2)
                input_lengths = torch.full(
                    (images.size(0),),
                    preds.size(1),
                    dtype=torch.long,
                )
                ctc_loss = self.criterion(preds_permuted, targets, input_lengths, target_lengths)

                # SR Loss (L1 between SR output and HR target)
                sr_loss = torch.tensor(0.0, device=self.device)
                if sr_output is not None:
                    # sr_output: [B, F, 3, H_sr, W_sr] in [0,1]
                    # hr_images: [B, F, 3, Hr, Wr] in [-1,1] (from val_transforms)
                    # Normalize hr to [0,1] to match sr_output
                    hr_01 = hr_images * 0.5 + 0.5

                    # Resize HR to match SR output spatial size if needed
                    b, f, c_ch, h_sr, w_sr = sr_output.shape
                    hr_flat = hr_01.view(b * f, c_ch, hr_01.shape[3], hr_01.shape[4])
                    sr_flat = sr_output.view(b * f, c_ch, h_sr, w_sr)

                    if hr_flat.shape[2:] != sr_flat.shape[2:]:
                        hr_flat = F.interpolate(
                            hr_flat, size=(h_sr, w_sr),
                            mode='bilinear', align_corners=False,
                        )

                    sr_loss = F.l1_loss(sr_flat, hr_flat)

                total_loss = ctc_loss + self.sr_loss_weight * sr_loss

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()

            epoch_loss += total_loss.item()
            epoch_ctc_loss += ctc_loss.item()
            epoch_sr_loss += sr_loss.item()

            pbar.set_postfix({
                'loss': total_loss.item(),
                'ctc': ctc_loss.item(),
                'sr': sr_loss.item(),
                'lr': self.scheduler.get_last_lr()[0],
            })

        n = len(self.train_loader)
        return {
            'loss':     epoch_loss / n,
            'ctc_loss': epoch_ctc_loss / n,
            'sr_loss':  epoch_sr_loss / n,
        }

    def validate(self) -> Tuple[Dict[str, float], List[str]]:
        """Run validation."""
        if self.val_loader is None:
            return {'loss': 0.0, 'acc': 0.0}, []

        self.model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        submission_data: List[str] = []

        with torch.no_grad():
            for batch in self.val_loader:
                images, targets, target_lengths, labels_text, track_ids, _ = batch
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Val: no SR loss, no return_sr needed
                preds = self.model(images, return_sr=False)

                input_lengths = torch.full(
                    (images.size(0),), preds.size(1), dtype=torch.long,
                )
                loss = self.criterion(
                    preds.permute(1, 0, 2), targets, input_lengths, target_lengths,
                )
                val_loss += loss.item()

                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    track_id = track_ids[i]
                    if pred_text == gt_text:
                        total_correct += 1
                    submission_data.append(f"{track_id},{pred_text};{conf:.4f}")

                total_samples += len(labels_text)

        avg_val_loss = val_loss / len(self.val_loader)
        val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0

        return {'loss': avg_val_loss, 'acc': val_acc}, submission_data

    def save_submission(self, submission_data: List[str]) -> None:
        exp_name = self._get_exp_name()
        filename = self._get_output_path(f"submission_{exp_name}.txt")
        with open(filename, 'w') as f:
            f.write("\n".join(submission_data))
        print(f"📝 Saved {len(submission_data)} lines to {filename}")

    def save_model(self, path: str = None) -> None:
        if path is None:
            exp_name = self._get_exp_name()
            path = self._get_output_path(f"{exp_name}_best.pth")
        torch.save(self.model.state_dict(), path)

    def fit(self) -> None:
        """Run the full training loop."""
        print(f"🚀 TRAINING START | Device: {self.device} | Epochs: {self.config.EPOCHS}")
        if self.use_sr:
            print(f"   SR BRANCH ENABLED | Loss weight: {self.sr_loss_weight}")

        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch

            train_metrics = self.train_one_epoch()
            val_metrics, submission_data = self.validate()

            val_acc = val_metrics['acc']
            current_lr = self.scheduler.get_last_lr()[0]

            sr_info = f" | SR Loss: {train_metrics['sr_loss']:.4f}" if self.use_sr else ""
            print(
                f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"CTC: {train_metrics['ctc_loss']:.4f}"
                f"{sr_info} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.2e}"
            )

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_model()
                exp_name = self._get_exp_name()
                model_path = self._get_output_path(f"{exp_name}_best.pth")
                print(f"  ⭐ Saved Best Model: {model_path} ({val_acc:.2f}%)")
                if submission_data:
                    self.save_submission(submission_data)

        if self.val_loader is None:
            self.save_model()

        print(f"\n✅ Training complete! Best Val Acc: {self.best_acc:.2f}%")

    def predict(self, loader: DataLoader) -> List[Tuple[str, str, float]]:
        """Run inference."""
        self.model.eval()
        results: List[Tuple[str, str, float]] = []

        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(self.device)
                track_ids = batch[4]

                preds = self.model(images, return_sr=False)
                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))

        return results

    def predict_test(self, test_loader: DataLoader, output_filename: str = "submission_final.txt") -> None:
        """Run inference on test data and save submission file."""
        print("🔮 Running inference on test data...")
        results = []
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test Inference"):
                images = batch[0].to(self.device)
                track_ids = batch[4]

                preds = self.model(images, return_sr=False)
                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))

        submission_data = [
            f"{track_id},{pred_text};{conf:.4f}"
            for track_id, pred_text, conf in results
        ]
        output_path = self._get_output_path(output_filename)
        with open(output_path, 'w') as f:
            f.write("\n".join(submission_data))

        print(f"✅ Saved {len(submission_data)} predictions to {output_path}")