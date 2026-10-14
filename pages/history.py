import html
import streamlit as st

from src import config
from src.services import runtime
from src.services.analysis import AnalysisResult
from src.storage.history import HistoryError
from src.ui.components import chip, render_result

PAGE_SIZE = 25  # number of records per page

st.markdown('<div class="dfd-brand">Deepfake Image Detector</div>', unsafe_allow_html=True)
st.title("History")
st.markdown(
    '<p class="dfd-lead">Saved analysis results and thumbnails stored locally.</p>',
    unsafe_allow_html=True,
)

store = runtime.get_store()  # get sqlite history store

if flash := st.session_state.pop("history_flash", None):
    st.success(flash)

total = store.count()  # total stored records
if total == 0:
    st.info("No analyses yet. Analyze an image and it will appear here.")
    st.stop()

top_left, top_right = st.columns([3, 1], vertical_alignment="center")
top_left.caption(f"{total} saved {'analysis' if total == 1 else 'analyses'}")

if not st.session_state.get("confirm_clear"):
    if top_right.button("Clear history", width="stretch"):
        st.session_state["confirm_clear"] = True  # show confirmation box
        st.rerun()
else:
    with st.container(border=True):
        st.warning(f"Delete all {total} saved records? This cannot be undone.")
        yes, no = st.columns(2)
        if yes.button("Yes, delete everything", type="primary", width="stretch"):
            store.clear(confirm=True)  # wipe database table
            st.session_state.pop("history_open", None)
            st.session_state.pop("confirm_clear", None)
            st.session_state["history_flash"] = "History cleared."
            st.rerun()
        if no.button("Cancel", width="stretch"):
            st.session_state.pop("confirm_clear", None)
            st.rerun()

open_id = st.session_state.get("history_open")  # currently expanded record
if open_id:
    try:
        rec = store.get(open_id)  # fetch single record
    except HistoryError:
        rec = None
    if rec is None:
        st.session_state.pop("history_open", None)
    else:
        with st.container(border=True):
            head_l, head_r = st.columns([4, 1], vertical_alignment="center")
            head_l.subheader(rec.filename or "Untitled image")
            if head_r.button("Close", width="stretch"):
                st.session_state.pop("history_open", None)
                st.rerun()
            if rec.thumbnail:
                st.image(rec.thumbnail)
            st.caption(f"Analyzed {rec.created_at.replace('T', ' ')} · {rec.width}x{rec.height}px · model {rec.model_id}")
            render_result(AnalysisResult.from_record(rec), runtime.get_model_card(rec.model_id))

shown = st.session_state.get("history_shown", PAGE_SIZE)  # pagination limit
for rec in store.list(limit=min(shown, 500)):
    with st.container(border=True):
        c_thumb, c_info, c_act = st.columns([1, 4, 2], vertical_alignment="center")
        if rec.thumbnail:
            c_thumb.image(rec.thumbnail, width="stretch")
        kind = "calibrated" if rec.score_is_calibrated else "raw score"
        label = {"likely_ai": "Likely AI-generated", "likely_real": "Likely real", "inconclusive": "Inconclusive"}[rec.decision]
        c_info.markdown(
            f"**{html.escape(rec.filename or 'Untitled image')}**<br>"
            f'<span class="dfd-muted">{rec.created_at.replace("T", " ")} · {rec.score * 100:.0f}% ({kind}) · '
            f"{html.escape(rec.model_id)}</span><br>{chip(rec.decision, label)}",
            unsafe_allow_html=True,
        )
        b1, b2 = c_act.columns(2)
        if b1.button("Open", key=f"open_{rec.id}", width="stretch"):
            st.session_state["history_open"] = rec.id  # open details
            st.rerun()
        if b2.button("Delete", key=f"del_{rec.id}", width="stretch"):
            store.delete(rec.id)  # delete this row
            if st.session_state.get("history_open") == rec.id:
                st.session_state.pop("history_open", None)
            st.rerun()

if total > shown:
    if st.button(f"Show more ({total - shown} remaining)"):
        st.session_state["history_shown"] = shown + PAGE_SIZE  # load more entries
