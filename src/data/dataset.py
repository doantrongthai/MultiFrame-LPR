"""MultiFrameDataset for license plate recognition with multi-frame input."""
import glob
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_degradation_transforms,
    get_light_transforms,
)


class MultiFrameDataset(Dataset):
    """Dataset for multi-frame license plate recognition.

    Handles both real LR images and synthetic LR (degraded HR) images.
    Implements Scenario-B specific validation splitting logic.

    When use_sr=True, real LR samples also return paired HR frames
    for SR supervision loss during end-to-end training.
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        split_ratio: float = 0.9,
        img_height: int = 32,
        img_width: int = 128,
        char2idx: Dict[str, int] = None,
        val_split_file: str = "data/val_tracks.json",
        seed: int = 42,
        augmentation_level: str = "full",
        is_test: bool = False,
        full_train: bool = False,
        use_sr: bool = False,           # NEW: return HR frames for SR loss
        hr_img_height: int = 128,       # NEW: HR target height (LR * scale)
        hr_img_width: int = 512,        # NEW: HR target width (LR * scale)
    ):
        """
        Args:
            root_dir: Root directory containing track folders.
            mode: 'train' or 'val'.
            split_ratio: Train/val split ratio.
            img_height: Target LR image height for OCR.
            img_width: Target LR image width for OCR.
            char2idx: Character to index mapping.
            val_split_file: Path to validation split JSON file.
            seed: Random seed for reproducible splitting.
            augmentation_level: 'full' or 'light' augmentation for training.
            is_test: If True, load test data without labels (for submission).
            full_train: If True, use all tracks for training (no val split).
            use_sr: If True, real LR samples also return HR frames for SR loss.
            hr_img_height: HR image resize height.
            hr_img_width: HR image resize width.
        """
        self.mode = mode
        self.samples: List[Dict[str, Any]] = []
        self.img_height = img_height
        self.img_width = img_width
        self.char2idx = char2idx or {}
        self.val_split_file = val_split_file
        self.seed = seed
        self.augmentation_level = augmentation_level
        self.is_test = is_test
        self.full_train = full_train
        self.use_sr = use_sr
        self.hr_img_height = hr_img_height
        self.hr_img_width = hr_img_width

        if mode == 'train':
            if augmentation_level == "light":
                self.transform = get_light_transforms(img_height, img_width)
            else:
                self.transform = get_train_transforms(img_height, img_width)
            self.degrade = get_degradation_transforms()
        else:
            self.transform = get_val_transforms(img_height, img_width)
            self.degrade = None

        # HR transform: only resize + normalize, no augmentation
        # Paired with LR so same deterministic processing
        self.hr_transform = get_val_transforms(hr_img_height, hr_img_width)

        print(f"[{mode.upper()}] Scanning: {root_dir}")
        abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))

        if not all_tracks:
            print("❌ ERROR: No data found.")
            return

        if is_test:
            print(f"[TEST] Loaded {len(all_tracks)} tracks.")
            self._index_test_samples(all_tracks)
            print(f"-> Total: {len(self.samples)} test samples.")
        else:
            train_tracks, val_tracks = self._load_or_create_split(all_tracks, split_ratio)
            selected_tracks = train_tracks if mode == 'train' else val_tracks
            print(f"[{mode.upper()}] Loaded {len(selected_tracks)} tracks.")
            self._index_samples(selected_tracks)
            print(f"-> Total: {len(self.samples)} samples.")

    def _load_or_create_split(
        self,
        all_tracks: List[str],
        split_ratio: float
    ) -> Tuple[List[str], List[str]]:
        """Load existing split or create new one with Scenario-B priority."""
        if self.full_train:
            print("📌 FULL TRAIN MODE: Using all tracks for training (no validation split).")
            return all_tracks, []

        train_tracks, val_tracks = [], []

        if os.path.exists(self.val_split_file):
            print(f"📂 Loading split from '{self.val_split_file}'...")
            try:
                with open(self.val_split_file, 'r') as f:
                    val_ids = set(json.load(f))
            except Exception:
                val_ids = set()

            for t in all_tracks:
                if os.path.basename(t) in val_ids:
                    val_tracks.append(t)
                else:
                    train_tracks.append(t)

            scenario_b_in_val = any("Scenario-B" in t for t in val_tracks)
            if not val_tracks or (not scenario_b_in_val and len(all_tracks) > 100):
                print("⚠️ Split file invalid or missing Scenario-B. Recreating...")
                val_tracks = []

        if not val_tracks:
            print("⚠️ Creating new split (Taking Val only from Scenario-B)...")
            scenario_b_tracks = [t for t in all_tracks if "Scenario-B" in t]
            if not scenario_b_tracks:
                print("⚠️ Warning: No 'Scenario-B' folder found. Using random from all.")
                scenario_b_tracks = all_tracks

            val_size = max(1, int(len(scenario_b_tracks) * (1 - split_ratio)))
            random.Random(self.seed).shuffle(scenario_b_tracks)
            val_tracks = scenario_b_tracks[:val_size]
            val_set = set(val_tracks)
            train_tracks = [t for t in all_tracks if t not in val_set]

            os.makedirs(os.path.dirname(self.val_split_file), exist_ok=True)
            with open(self.val_split_file, 'w') as f:
                json.dump([os.path.basename(t) for t in val_tracks], f, indent=2)

        return train_tracks, val_tracks

    def _index_samples(self, tracks: List[str]) -> None:
        """Index all samples from selected tracks."""
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label:
                    continue

                track_id = os.path.basename(track_path)

                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png")) +
                    glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                hr_files = sorted(
                    glob.glob(os.path.join(track_path, "hr-*.png")) +
                    glob.glob(os.path.join(track_path, "hr-*.jpg"))
                )

                # Real LR samples — include hr_paths for SR supervision if use_sr
                self.samples.append({
                    'paths': lr_files,
                    'hr_paths': hr_files if (self.use_sr and self.mode == 'train') else [],
                    'label': label,
                    'is_synthetic': False,
                    'track_id': track_id,
                })

                # Synthetic LR samples (only in training mode, no HR needed — HR IS the input)
                if self.mode == 'train':
                    self.samples.append({
                        'paths': hr_files,
                        'hr_paths': [],  # Synthetic: HR degraded → no separate HR target
                        'label': label,
                        'is_synthetic': True,
                        'track_id': track_id,
                    })
            except Exception:
                pass

    def _index_test_samples(self, tracks: List[str]) -> None:
        """Index test samples without labels."""
        for track_path in tqdm(tracks, desc="Indexing test"):
            track_id = os.path.basename(track_path)
            lr_files = sorted(
                glob.glob(os.path.join(track_path, "lr-*.png")) +
                glob.glob(os.path.join(track_path, "lr-*.jpg"))
            )
            if lr_files:
                self.samples.append({
                    'paths': lr_files,
                    'hr_paths': [],
                    'label': '',
                    'is_synthetic': False,
                    'track_id': track_id,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[
        torch.Tensor,   # LR images [F, C, H, W]
        torch.Tensor,   # targets
        int,            # target_len
        str,            # label text
        str,            # track_id
        torch.Tensor,   # HR images [F, C, Hr, Wr] or empty tensor
    ]:
        """Load frames. Returns HR frames if use_sr=True and available."""
        item = self.samples[idx]
        img_paths = item['paths']
        hr_paths  = item['hr_paths']
        label = item['label']
        is_synthetic = item['is_synthetic']
        track_id = item['track_id']

        # --- Load LR frames ---
        images_list = []
        for p in img_paths:
            image = cv2.imread(p, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if is_synthetic and self.degrade:
                image = self.degrade(image=image)['image']

            image = self.transform(image=image)['image']
            images_list.append(image)

        images_tensor = torch.stack(images_list, dim=0)  # [F, C, H, W]

        # --- Load HR frames (for SR supervision) ---
        hr_tensor = torch.empty(0)  # Default: empty if not needed
        if hr_paths and self.use_sr and self.mode == 'train':
            hr_list = []
            for p in hr_paths:
                hr_img = cv2.imread(p, cv2.IMREAD_COLOR)
                hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
                hr_img = self.hr_transform(image=hr_img)['image']
                hr_list.append(hr_img)
            if hr_list:
                hr_tensor = torch.stack(hr_list, dim=0)  # [F, C, Hr, Wr]

        # --- Targets ---
        if self.is_test:
            target = [0]
            target_len = 1
        else:
            target = [self.char2idx[c] for c in label if c in self.char2idx]
            if len(target) == 0:
                target = [0]
            target_len = len(target)

        return (
            images_tensor,
            torch.tensor(target, dtype=torch.long),
            target_len,
            label,
            track_id,
            hr_tensor,
        )

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple:
        """Custom collate function for DataLoader."""
        images, targets, target_lengths, labels_text, track_ids, hr_images = zip(*batch)

        images = torch.stack(images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)

        # HR images: stack only if all non-empty, else return empty tensor
        has_hr = all(h.numel() > 0 for h in hr_images)
        if has_hr:
            hr_images_tensor = torch.stack(hr_images, 0)
        else:
            hr_images_tensor = torch.empty(0)

        return images, targets, target_lengths, labels_text, track_ids, hr_images_tensor