"""Detector registry coverage."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection.models.registry import DETECTOR_REGISTRY, get_detector
from anomaly_detection.multimodal.fusion import MultimodalFusion


def test_multimodal_fusion_registered() -> None:
    assert "multimodal_fusion" in DETECTOR_REGISTRY
    assert DETECTOR_REGISTRY["multimodal_fusion"] is MultimodalFusion


def test_get_detector_multimodal_fusion_roundtrip() -> None:
    detector = get_detector("multimodal_fusion", {"contamination": 0.1, "latent_dim": 2})
    assert isinstance(detector, MultimodalFusion)
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [10.0, 10.0]])
    texts = ["ok", "ok", "ok", "spike anomaly"]
    detector.fit(X, texts=texts)
    preds = detector.predict(X, texts=texts)
    assert preds.shape == (4,)


def test_unknown_detector_lists_multimodal() -> None:
    with pytest.raises(KeyError, match="multimodal_fusion"):
        get_detector("not-a-real-detector")
