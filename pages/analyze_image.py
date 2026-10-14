import hashlib
import streamlit as st

from src.data import imageops
from src.services import runtime
from src.services.analysis import InferenceError, analyze_upload
from src.ui.components import render_how_it_works, render_result

st.markdown('<div class="dfd-brand">Deepfake Image Detector</div>', unsafe_allow_html=True)
st.title("Is this image AI-generated?")
st.markdown(
    '<p class="dfd-lead">Upload an image to estimate whether it is likely real or AI-generated.</p>',
    unsafe_allow_html=True,
)

predictor, load_error = runtime.get_predictor()  # load trained model
if predictor is None:
    st.error(f"The detection model is not available. {load_error}")

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False,
    help="JPEG, PNG or WebP up to 15MB",
)

if uploaded is None:
    st.session_state.pop("current_result", None)  # reset result if no file
else:
    data = uploaded.getvalue()  # read uploaded bytes
    file_key = hashlib.sha256(data).hexdigest()  # hash to track file changes

    try:
        image, info = imageops.inspect_bytes(data)  # validate image format and size
    except imageops.InvalidImageError as exc:
        st.error(exc.message)
        st.session_state.pop("current_result", None)
    else:
        if st.session_state.get("current_result_key") != file_key:
            st.session_state.pop("current_result", None)  # new image uploaded, clear previous output

        left, right = st.columns([3, 2], vertical_alignment="center")
        with left:
            st.image(imageops.to_rgb(image), caption=f"{uploaded.name} ({info.width}x{info.height}px)", width="stretch")
        with right:
            analyze = st.button("Analyze Image", type="primary", disabled=predictor is None, width="stretch")  # trigger inference
            st.caption("Runs locally on your device. Results are saved to history.")

        if analyze and predictor is not None:
            try:
                with st.spinner("Analyzing image..."):
                    result = analyze_upload(data, uploaded.name, predictor, runtime.get_store())  # run prediction
                st.session_state["current_result"] = result
                st.session_state["current_result_key"] = file_key
            except imageops.InvalidImageError as exc:
                st.error(exc.message)
            except InferenceError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"An error occurred while analyzing image: {exc}")

result = st.session_state.get("current_result")
if uploaded is not None and result is not None:
    card = runtime.get_model_card(result.model_id)  # fetch model info
    render_result(result, card)  # display prediction details

render_how_it_works(
    predictor.meta if predictor else None,
    runtime.get_model_card(predictor.model_id) if predictor else {"available": False},
    predictor.calibration_info if predictor else None,
)
