from __future__ import annotations

import io
import random

import numpy as np
import torch
from PIL import Image, ImageFilter

from src.config import CANVAS_SIZE, CROP_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # default normalization mean
IMAGENET_STD = (0.229, 0.224, 0.225)   # default normalization std


def to_normalized_tensor(img: Image.Image, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # scale pixel values to 0-1
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # hwc to chw
    return (tensor - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)  # normalize


def center_crop(img: Image.Image, size: int = CROP_SIZE) -> Image.Image:
    width, height = img.size
    left, top = (width - size) // 2, (height - size) // 2
    return img.crop((left, top, left + size, top + size))  # take center square crop


def eval_tensor(canvas: Image.Image, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    assert canvas.size == (CANVAS_SIZE, CANVAS_SIZE), f"expected canvas {CANVAS_SIZE}px, got {canvas.size}"
    return to_normalized_tensor(center_crop(canvas), mean, std)  # center crop and normalize


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)  # compress to jpeg in memory
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class TrainTransform:
    def __init__(
        self,
        rescale_prob=0.30,
        jpeg_prob=0.40,
        blur_prob=0.15,
        flip=True,
        rescale_range=(0.6, 1.4),
        jpeg_range=(55, 98),
        blur_range=(0.3, 1.5),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ):
        self.rescale_prob, self.jpeg_prob, self.blur_prob, self.flip = rescale_prob, jpeg_prob, blur_prob, flip
        self.rescale_range, self.jpeg_range, self.blur_range = rescale_range, jpeg_range, blur_range
        self.mean, self.std = mean, std

    def __call__(self, canvas: Image.Image) -> torch.Tensor:
        img = canvas
        if random.random() < self.rescale_prob:
            size = max(CROP_SIZE, int(round(CANVAS_SIZE * random.uniform(*self.rescale_range))))
            img = img.resize((size, size), Image.BICUBIC)  # random scaling
        if random.random() < self.blur_prob:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(*self.blur_range)))  # random blur
        if random.random() < self.jpeg_prob:
            img = jpeg_roundtrip(img, random.randint(*self.jpeg_range))  # random compression

        width, height = img.size
        left = random.randint(0, width - CROP_SIZE)  # random crop
        top = random.randint(0, height - CROP_SIZE)
        img = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

        if self.flip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip

        return to_normalized_tensor(img, self.mean, self.std)
