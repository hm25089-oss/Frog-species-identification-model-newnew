import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from tensorflow import keras

from train_cnn import (
    MODEL_DIR,
    MODEL_NAME,
    extract_segments,
)

@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_DIR / f"{MODEL_NAME}.keras")
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    normalizer = np.load(MODEL_DIR / "normalizer.npz")
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    return model, encoder, normalizer, metadata


def predict(path: Path, model, encoder, normalizer):
    segments = np.array(extract_segments(path))
    scaled = (segments - normalizer["mean"]) / normalizer["std"]
    probabilities = model.predict(scaled, verbose=0).mean(axis=0)

    rows = [
        {"종류": label, "확률(%)": round(float(probability * 100), 1)}
        for label, probability in zip(encoder.classes_, probabilities)
    ]
    return sorted(rows, key=lambda row: row["확률(%)"], reverse=True)


st.set_page_config(page_title="개구리 CNN 분류", layout="centered")
st.title(f"개구리 종 소리 분류 ({MODEL_NAME})")

try:
    model, encoder, normalizer, metadata = load_artifacts()
except Exception as exc:
    st.error(f"새 모델을 찾지 못했습니다. train_cnn.py를 먼저 실행하세요: {exc}")
    st.stop()

uploaded = st.file_uploader(
    "음원 선택",
    type=["wav", "mp3", "flac", "ogg", "m4a", "aac"],
)

if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(uploaded.getbuffer())
        path = Path(temp.name)
    try:
        rows = predict(path, model, encoder, normalizer)
        st.subheader(f"예측: {rows[0]['종류']} ({rows[0]['확률(%)']}%)")
        st.dataframe(rows, hide_index=True)
    finally:
        path.unlink(missing_ok=True)

accuracy = metadata.get("training_accuracy", 0) * 100
st.caption(f"학습 데이터 기준 정확도: {accuracy:.1f}% — 새 녹음 성능은 아닙니다.")
