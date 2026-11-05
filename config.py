from pathlib import Path

# 바탕화면/음향 폴더를 자동으로 찾습니다.
# 다른 위치에 있다면 아래 AUDIO_DIR만 직접 수정하세요.
# 예: AUDIO_DIR = Path(r"D:\데이터\음향")
AUDIO_DIR = None

MODEL_DIR = Path(__file__).parent / "model"
SAMPLE_RATE = 44100
N_MFCC = 40
EPOCHS = 50
BATCH_SIZE = 4
RANDOM_STATE = 42

# 폴더 이름 = 정답 라벨. 실제 학습에는 음원이 있는 폴더만 포함됩니다.
EXPECTED_FOLDERS = [
    "두꺼비",
    "맹꽁이",
    "배경소리",
    "북방산개구리",
    "참개구리",
    "청개구리",
    "황소개구리",
]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
