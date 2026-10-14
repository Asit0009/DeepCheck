import io

import numpy as np
import pytest
from PIL import Image

from src.data import imageops
from src.data.imageops import InvalidImageError
from tests.conftest import make_jpeg, make_png, make_webp


@pytest.mark.parametrize("maker,fmt", [(make_png, "PNG"), (make_jpeg, "JPEG"), (make_webp, "WEBP")])
def test_supported_formats_decode(maker, fmt):
    img, info = imageops.inspect_bytes(maker())
    assert info.format == fmt
    assert img.size == (info.width, info.height)


def test_jpeg_quality_is_estimated():
    _, info = imageops.inspect_bytes(make_jpeg(quality=90))
    assert info.jpeg_quality is not None and 85 <= info.jpeg_quality <= 95


def test_empty_file_is_rejected():
    with pytest.raises(InvalidImageError) as exc:
        imageops.inspect_bytes(b"")
    assert exc.value.code == "empty"


def test_random_bytes_are_not_an_image():
    with pytest.raises(InvalidImageError) as exc:
        imageops.inspect_bytes(b"this is definitely not an image" * 50)
    assert exc.value.code == "not_an_image"


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_unsupported_formats_are_rejected(fmt):
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), (10, 20, 30)).save(buf, format=fmt)
    with pytest.raises(InvalidImageError) as exc:
        imageops.inspect_bytes(buf.getvalue())
    assert exc.value.code == "unsupported_format"
    assert "JPEG, PNG or WebP" in exc.value.message


def test_truncated_file_is_reported_as_corrupt():
    data = make_jpeg(size=(400, 400))
    with pytest.raises(InvalidImageError) as exc:
        imageops.inspect_bytes(data[: len(data) // 2])
    assert exc.value.code in {"corrupt", "not_an_image"}


def test_file_size_limit():
    with pytest.raises(InvalidImageError) as exc:
        imageops.inspect_bytes(make_png(), max_bytes=1000)
    assert exc.value.code == "too_large"


def test_dimension_minimum():
    with pytest.raises(InvalidImageError) as exc:
        imageops.inspect_bytes(make_png(size=(40, 300)))
    assert exc.value.code == "too_small"


def test_decompression_bomb_is_refused(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 5000)
    with pytest.raises(InvalidImageError):
        imageops.inspect_bytes(make_png(size=(300, 300)))


def test_transparency_is_flattened_onto_white():
    rgba = Image.new("RGBA", (80, 80), (255, 0, 0, 0))     # fully transparent red
    out = imageops.to_rgb(rgba)
    assert out.mode == "RGB" and out.getpixel((10, 10)) == (255, 255, 255)


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (120, 80), (0, 0, 0))
    exif = img.getexif()
    exif[0x0112] = 6                                       # rotate 90 degrees
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    decoded, _ = imageops.inspect_bytes(buf.getvalue())
    assert imageops.to_rgb(decoded).size == (80, 120)


@pytest.mark.parametrize("size", [(300, 200), (200, 300), (256, 256), (128, 128), (900, 700)])
def test_canonical_output_is_square_256(size):
    img, _ = imageops.inspect_bytes(make_png(size=size))
    canvas = imageops.canonicalize(img)
    assert canvas.size == (256, 256) and canvas.mode == "RGB"


def test_canonicalisation_is_deterministic():
    img, _ = imageops.inspect_bytes(make_png(size=(321, 213)))
    assert np.array_equal(np.asarray(imageops.canonicalize(img)), np.asarray(imageops.canonicalize(img)))


def test_identical_images_share_hashes_and_different_images_do_not():
    a, _ = imageops.inspect_bytes(make_png(seed=1))
    b, _ = imageops.inspect_bytes(make_png(seed=1))
    c, _ = imageops.inspect_bytes(make_png(seed=2, color=(10, 200, 30)))
    ca, cb, cc = (imageops.canonicalize(x) for x in (a, b, c))
    assert imageops.phash64(ca) == imageops.phash64(cb) and imageops.pixel_sha256(a) == imageops.pixel_sha256(b)
    assert bin(imageops.phash64(ca) ^ imageops.phash64(cc)).count("1") > 0 or imageops.pixel_sha256(a) != imageops.pixel_sha256(c)
