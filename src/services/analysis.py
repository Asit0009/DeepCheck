from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import datetime, timezone

from PIL import Image

from src.data import imageops
from src.models.predictor import LABELS, Prediction, Predictor
from src.storage.history import HistoryRecord, HistoryStore, sanitize_filename

THUMBNAIL_SIZE = 160  # max thumbnail dimension in pixels


class InferenceError(RuntimeError):
    pass


@dataclass
class AnalysisResult:
    id: str | None
    created_at: str
    filename: str | None
    image_sha256: str
    width: int | None
    height: int | None
    image_format: str | None
    decision: str
    score: float
    score_is_calibrated: bool
    model_id: str
    model_version: str
    latency_ms: float | None
    thumbnail: bytes | None
    saved_to_history: bool = False

    @property
    def label(self) -> str:
        return LABELS[self.decision]

    @classmethod
    def from_record(cls, rec: HistoryRecord) -> "AnalysisResult":
        return cls(
            id=rec.id,
            created_at=rec.created_at,
            filename=rec.filename,
            image_sha256=rec.image_sha256,
            width=rec.width,
            height=rec.height,
            image_format=rec.image_format,
            decision=rec.decision,
            score=rec.score,
            score_is_calibrated=rec.score_is_calibrated,
            model_id=rec.model_id,
            model_version=rec.model_version,
            latency_ms=rec.latency_ms,
            thumbnail=rec.thumbnail,
            saved_to_history=True,
        )


def make_thumbnail(img: Image.Image, size: int = THUMBNAIL_SIZE) -> bytes:
    thumb = imageops.to_rgb(img)
    thumb.thumbnail((size, size))  # resize to thumbnail scale
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=70)  # save compressed jpeg
    return buf.getvalue()


def analyze_upload(
    data: bytes,
    filename: str | None,
    predictor: Predictor,
    store: HistoryStore | None = None,
    keep_thumbnail: bool = True,
) -> AnalysisResult:
    img, info = imageops.inspect_bytes(data)  # decode and inspect image bytes
    try:
        pred: Prediction = predictor.predict_image(img)  # execute model inference
    except Exception as exc:
        raise InferenceError(f"The model could not analyse this image: {exc}") from exc

    sha = hashlib.sha256(data).hexdigest()  # compute sha256 checksum
    thumb = make_thumbnail(img) if keep_thumbnail else None

    result = AnalysisResult(
        id=None,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        filename=sanitize_filename(filename),
        image_sha256=sha,
        width=info.width,
        height=info.height,
        image_format=info.format,
        decision=pred.decision,
        score=pred.score,
        score_is_calibrated=pred.score_is_calibrated,
        model_id=pred.model_id,
        model_version=pred.model_version,
        latency_ms=pred.latency_ms,
        thumbnail=thumb,
    )

    if store is not None:
        result.id = store.add(
            filename=result.filename,
            image_sha256=sha,
            decision=result.decision,
            score=result.score,
            score_is_calibrated=result.score_is_calibrated,
            logit=pred.logit,
            model_id=result.model_id,
            model_version=result.model_version,
            latency_ms=result.latency_ms,
            width=info.width,
            height=info.height,
            image_format=info.format,
            thumbnail=thumb,
        )  # save entry in database
        result.saved_to_history = True

    return result
