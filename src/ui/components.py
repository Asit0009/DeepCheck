from __future__ import annotations

import html
import streamlit as st

from src.services.analysis import AnalysisResult

EXPLANATIONS = {
    "likely_ai": "The model detected patterns common in AI-generated images.",
    "likely_real": "The model did not find strong indicators of AI generation.",
    "inconclusive": "The score is in the uncertain middle range, so the model cannot determine with confidence.",
}

UNCERTAINTY_WARNING = (
    "A model score cannot establish where an image came from. Compression, resizing, screenshots, editing, and "
    "images from generators newer than the training data can be misjudged in either direction."
)


def render_result(result: AnalysisResult, card: dict) -> None:
    pct = result.score * 100  # score as percentage
    st.markdown(
        f'<div class="dfd-panel">'
        f'<span class="dfd-chip {result.decision}">{html.escape(result.label)}</span>'
        f'<div class="dfd-score">{pct:.0f}%</div>'
        f'<div class="dfd-meter"><div style="width:{pct:.1f}%"></div></div>',
        unsafe_allow_html=True,
    )
    if result.score_is_calibrated:
        meaning = (
            "Estimated probability that this image is AI-generated. The score was calibrated on held-out "
            "images from the generators the model was trained on, so it is most trustworthy for similar images."
        )
    else:
        meaning = (
            'Raw model score for "AI-generated". It has NOT been calibrated, so it is not a verified '
            "probability. Higher means the model sees more AI-like patterns."
        )

    st.markdown(f'<div class="dfd-muted">{html.escape(meaning)}</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="margin-top:.9rem">{html.escape(EXPLANATIONS[result.decision])}</p></div>', unsafe_allow_html=True)

    if card.get("available"):
        extra = ""
        if "unseen_balanced_accuracy" in card:
            extra = (
                f'<span class="dfd-stat"><b>{card["unseen_balanced_accuracy"] * 100:.1f}%</b>'
                f'<span class="dfd-muted">accuracy on unseen generators</span></span>'
            )
        st.markdown(
            f'<div class="dfd-panel"><div class="dfd-muted" style="margin-bottom:.5rem">'
            f'Model performance benchmark ({card["test_n"]:,} test samples):</div>'
            f'<span class="dfd-stat"><b>{card["test_accuracy"] * 100:.1f}%</b><span class="dfd-muted">accuracy</span></span>'
            f'<span class="dfd-stat"><b>{card["test_macro_f1"]:.3f}</b><span class="dfd-muted">macro-F1</span></span>'
            f"{extra}</div>",
            unsafe_allow_html=True,
        )
    st.markdown(f'<div class="dfd-warn">{html.escape(UNCERTAINTY_WARNING)}</div>', unsafe_allow_html=True)


def render_how_it_works(predictor_meta: dict | None, card: dict, calibration_info: dict | None) -> None:
    with st.expander("How does this work?"):
        meta = predictor_meta or {}
        arch = html.escape(str(meta.get("timm_name", "image classifier")))
        st.markdown(
            f"""
1. **Image Preprocessing:** The uploaded image is verified, centered, and scaled to 256x256.
2. **Feature Extraction:** A deep neural network (`{arch}`) extracts visual feature representations.
3. **Probability Scoring:** The classifier outputs a score and maps it using calibration cutoffs.
"""
        )
        if card.get("available"):
            st.caption(f"Model: {meta.get('model_id', 'default')} · test accuracy: {card.get('test_accuracy', 0) * 100:.1f}%")


def chip(decision: str, label: str) -> str:
    return f'<span class="dfd-chip {decision}" style="font-size:.85rem;padding:.2rem .65rem">{html.escape(label)}</span>'
