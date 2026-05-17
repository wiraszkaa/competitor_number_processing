"""Fine-tune EasyOCR's recognition model on labeled bib crops."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from pipeline.config import get_pipeline_logger

logger = get_pipeline_logger(__name__)

_IMG_HEIGHT = 64  # EasyOCR's expected input height


class BibCropDataset(Dataset):
    def __init__(self, csv_path: Path, crop_dir: Path):
        self.crop_dir = crop_dir
        self.samples: List[Tuple[Path, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label = row.get("correct_number", "").strip()
                if label:
                    self.samples.append((crop_dir / row["file"], label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")  # grayscale
        w, h = img.size
        new_w = max(1, int(w * _IMG_HEIGHT / h))
        img = img.resize((new_w, _IMG_HEIGHT), Image.BICUBIC)
        tensor = torch.tensor(list(img.getdata()), dtype=torch.float32)
        tensor = tensor.view(1, _IMG_HEIGHT, new_w) / 255.0
        tensor.sub_(0.5).div_(0.5)
        return tensor, label


def _collate(batch: List[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, List[str]]:
    images, labels = zip(*batch)
    max_w = max(t.shape[2] for t in images)
    padded = torch.zeros(len(images), 1, _IMG_HEIGHT, max_w)
    for i, t in enumerate(images):
        padded[i, :, :, : t.shape[2]] = t
    return padded, list(labels)


def fine_tune_ocr(
    csv_path: Path,
    crop_dir: Path,
    output_path: Path,
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 16,
) -> None:
    """Fine-tune EasyOCR's recognition model on labeled bib crops via CTC loss."""
    import easyocr

    dataset = BibCropDataset(csv_path, crop_dir)
    if len(dataset) == 0:
        logger.warning("No labeled rows in labels.csv — skipping OCR fine-tuning")
        return

    logger.info(f"OCR fine-tuning: {len(dataset)} labeled crops, {epochs} epochs")

    reader = easyocr.Reader(["en"], gpu=False, verbose=False, quantize=False)
    model = reader.recognizer
    converter = reader.converter

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for images, labels in loader:
            text, text_lengths = converter.encode(labels)
            preds = model(images, None)               # (N, T, C) — text unused in CTC models
            preds = preds.permute(1, 0, 2)            # (T, N, C) for CTCLoss
            T, N = preds.size(0), preds.size(1)
            preds_size = torch.IntTensor([T] * N)

            loss = ctc_loss(
                preds.log_softmax(2).float(),
                text,
                preds_size,
                text_lengths,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        logger.info(f"  Epoch {epoch:2d}/{epochs}  CTC loss: {avg:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info(f"[OK] Fine-tuned OCR weights saved to {output_path}")
