"""
app.py — Streamlit UI for NSFW audio keyword-spotting via Acoustic Word
Embeddings (no ASR / full transcription involved).

Run with:
    streamlit run app.py
"""
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import plotly.graph_objects as go

import config as cfg
from features import load_audio
from matcher import load_model_and_bank, run_pipeline_with_loaded, Detection

st.set_page_config(page_title="NSFW Audio Keyword Spotter", page_icon="🔊", layout="wide")


@st.cache_resource(show_spinner="Loading AWE model and reference bank...")
def get_model_and_bank():
    return load_model_and_bank()


def check_assets_ready() -> list[str]:
    problems = []
    if not cfg.AWE_MODEL_PATH.exists():
        problems.append(f"Trained AWE model not found at `{cfg.AWE_MODEL_PATH}`. "
                         f"Run `train_awe.py` first.")
    if not cfg.REFERENCE_BANK_PATH.exists():
        problems.append(f"Reference bank not found at `{cfg.REFERENCE_BANK_PATH}`. "
                         f"Run `build_reference_bank.py` first.")
    return problems


def plot_waveform(y: np.ndarray, sr: int, detections: list[Detection]):
    t = np.arange(len(y)) / sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="waveform",
                              line=dict(color="#888", width=1)))
    for det in detections:
        fig.add_vrect(
            x0=det.start_sec, x1=det.end_sec,
            fillcolor="red", opacity=0.25, line_width=0,
            annotation_text=f"{det.word} ({det.score:.2f})",
            annotation_position="top left",
        )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                       xaxis_title="Time (s)", yaxis_title="Amplitude",
                       showlegend=False)
    return fig


def main():
    st.title("🔊 NSFW Audio Keyword Spotter")
    st.caption(
        "Detects NSFW keywords in uploaded audio using **Acoustic Word Embeddings (AWE)** "
        "similarity matching — no ASR / full speech-to-text is used at any point."
    )

    problems = check_assets_ready()
    if problems:
        st.error("Setup incomplete:")
        for p in problems:
            st.markdown(f"- {p}")
        st.info(
            "Pipeline order: `generate_references.py` → `train_awe.py` → "
            "`build_reference_bank.py` → `streamlit run app.py`"
        )
        st.stop()

    model, device, bank_words, bank_matrix = get_model_and_bank()
    unique_words = sorted(set(bank_words))

    with st.sidebar:
        st.header("Settings")
        threshold = st.slider(
            "Similarity threshold", min_value=0.0, max_value=1.0,
            value=float(cfg.DEFAULT_SIM_THRESHOLD), step=0.01,
            help="Higher = fewer false positives but may miss some words. "
                 "See checkpoints/threshold_report.json for a calibrated suggestion."
        )
        st.markdown("---")
        st.markdown(f"**Loaded NSFW words:** {len(unique_words)}")
        with st.expander("Show word list (reference bank)"):
            st.write(unique_words)
        st.markdown("---")
        st.caption(f"Device: `{device}`")

    uploaded = st.file_uploader(
        "Upload an audio file", type=["wav", "mp3", "m4a", "flac", "ogg"]
    )

    if uploaded is None:
        st.info("Upload a voice file to scan it for NSFW keywords.")
        return

    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    y = load_audio(tmp_path, sr=cfg.SAMPLE_RATE)
    st.audio(uploaded)

    with st.spinner("Scanning audio for NSFW keywords..."):
        detections = run_pipeline_with_loaded(
            model, device, bank_words, bank_matrix, y, threshold=threshold
        )

    st.markdown("---")
    if detections:
        st.error(f"🚫 NSFW content detected — {len(detections)} keyword occurrence(s) found.")
    else:
        st.success("✅ No NSFW keywords detected above the current threshold.")

    st.plotly_chart(plot_waveform(y, cfg.SAMPLE_RATE, detections), use_container_width=True)

    if detections:
        st.subheader("Detected occurrences")
        for i, det in enumerate(detections):
            cols = st.columns([2, 2, 2, 3])
            cols[0].markdown(f"**Word:** `{det.word}`")
            cols[1].markdown(f"**Confidence:** {det.score:.3f}")
            cols[2].markdown(f"**Time:** {det.start_sec:.2f}s – {det.end_sec:.2f}s")

            clip = y[int(det.start_sec * cfg.SAMPLE_RATE):int(det.end_sec * cfg.SAMPLE_RATE)]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as clip_f:
                sf.write(clip_f.name, clip, cfg.SAMPLE_RATE)
                cols[3].audio(clip_f.name)


if __name__ == "__main__":
    main()
