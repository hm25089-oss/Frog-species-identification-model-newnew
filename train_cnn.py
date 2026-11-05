import json
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

from config import AUDIO_DIR, AUDIO_EXTENSIONS, EXPECTED_FOLDERS

SAMPLE_RATE = 44100
N_MFCC = 40
CLIP_SECONDS = 3
HOP_SECONDS = 1.5
TIME_STEPS = 130
EPOCHS = 40
BATCH_SIZE = 16
RANDOM_STATE = 42

MODEL_NAME = "frog_cnn_v2"
MODEL_DIR = Path(__file__).parent / "models" / MODEL_NAME


def find_audio_dir() -> Path:
    if AUDIO_DIR is not None:
        path = Path(AUDIO_DIR)
        if path.is_dir():
            return path
        raise FileNotFoundError(f"음향 폴더가 없습니다: {path}")

    candidates = [
        Path.home() / "Desktop" / "음향",
        Path.home() / "바탕 화면" / "음향",
        Path.home() / "OneDrive" / "Desktop" / "음향",
        Path.home() / "OneDrive" / "바탕 화면" / "음향",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError("바탕화면의 '음향' 폴더를 찾지 못했습니다.")


def extract_segments(path: Path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    clip_length = SAMPLE_RATE * CLIP_SECONDS
    hop_length = int(SAMPLE_RATE * HOP_SECONDS)

    if len(audio) < clip_length:
        audio = np.pad(audio, (0, clip_length - len(audio)))

    segments = []
    for start in range(0, len(audio) - clip_length + 1, hop_length):
        clip = audio[start:start + clip_length]
        mfcc = librosa.feature.mfcc(
            y=clip,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
        )
        mfcc = mfcc[:, :TIME_STEPS]
        if mfcc.shape[1] < TIME_STEPS:
            mfcc = np.pad(mfcc, ((0, 0), (0, TIME_STEPS - mfcc.shape[1])))
        segments.append(mfcc[..., np.newaxis].astype(np.float32))
    return segments


def load_dataset(audio_dir: Path):
    x, labels = [], []
    counts = {}

    for folder in sorted(path for path in audio_dir.iterdir() if path.is_dir()):
        files = [
            path for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ]
        if not files:
            continue

        counts[folder.name] = 0
        for file_path in files:
            try:
                segments = extract_segments(file_path)
            except Exception as exc:
                print(f"[실패] {file_path.name}: {exc}")
                continue
            x.extend(segments)
            labels.extend([folder.name] * len(segments))
            counts[folder.name] += 1
            print(f"읽음: {folder.name}/{file_path.name} → 구간 {len(segments)}개")

    missing = [name for name in EXPECTED_FOLDERS if counts.get(name, 0) == 0]
    if missing:
        print("음원이 없는 폴더:", ", ".join(missing))
    if len(set(labels)) < 2:
        raise RuntimeError("최소 2개 카테고리의 음원이 필요합니다.")
    return np.array(x), np.array(labels), counts


def build_model(num_classes: int):
    model = keras.Sequential([
        layers.Input(shape=(N_MFCC, TIME_STEPS, 1)),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPool2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    audio_dir = find_audio_dir()
    x, labels, counts = load_dataset(audio_dir)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    mean = x.mean(axis=(0, 1, 2), keepdims=True)
    std = x.std(axis=(0, 1, 2), keepdims=True) + 1e-6
    x_scaled = (x - mean) / std

    keras.utils.set_random_seed(RANDOM_STATE)
    model = build_model(len(encoder.classes_))
    history = model.fit(
        x_scaled,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=1,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / f"{MODEL_NAME}.keras")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    np.savez(MODEL_DIR / "normalizer.npz", mean=mean, std=std)

    metadata = {
        "model_name": MODEL_NAME,
        "sample_counts": counts,
        "segment_count": int(len(y)),
        "classes": encoder.classes_.tolist(),
        "training_accuracy": float(history.history["accuracy"][-1]),
        "input_shape": [N_MFCC, TIME_STEPS, 1],
        "clip_seconds": CLIP_SECONDS,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n저장 완료: {MODEL_DIR}")


if __name__ == "__main__":
    main()
