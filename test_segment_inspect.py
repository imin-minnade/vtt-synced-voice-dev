"""WhisperX の segment 単位での分割結果を確認するスクリプト。

segment をそのまま1キューとして扱った場合の粒度を表示する。
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import whisperx

SAMPLE_RATE = 16000
AUDIO_FILE = "audio_input/audio1697945985.m4a"

# --- 音声抽出 ---
audio_path = Path(AUDIO_FILE)
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    wav_path = tmp.name

subprocess.run(
    ["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", str(SAMPLE_RATE), wav_path],
    capture_output=True,
)
print(f"音声抽出完了: {wav_path}")

# --- WhisperX 実行 ---
model = whisperx.load_model("large-v2", "cpu", language="ja")
audio = whisperx.load_audio(wav_path)
result = model.transcribe(audio, language="ja")
segments = result["segments"]

model_a, metadata = whisperx.load_align_model(language_code="ja", device="cpu")
aligned = whisperx.align(segments, model_a, metadata, audio, "cpu", return_char_alignments=False)
aligned_segments = aligned["segments"]

print(f"\n=== WhisperX segment 単位での分割結果 ({len(aligned_segments)} segments) ===\n")

def fmt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

for i, seg in enumerate(aligned_segments):
    start = seg.get("start", 0)
    end = seg.get("end", 0)
    text = seg.get("text", "").strip()
    words = seg.get("words", [])
    n_words = len(words)
    n_chars = len(text)
    print(f"[{i:3d}] {fmt(start)} --> {fmt(end)}  ({n_chars}文字/{n_words}単語)")
    print(f"      「{text}」")

Path(wav_path).unlink(missing_ok=True)
