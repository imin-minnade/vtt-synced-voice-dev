"""segment 内の単語間ギャップ分布を分析し、適切な分割閾値を探るスクリプト。"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import whisperx

SAMPLE_RATE = 16000
AUDIO_FILE = "audio_input/audio1697945985.m4a"
SENTENCE_END_DURATION = 0.15

def fmt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

audio_path = Path(AUDIO_FILE)
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    wav_path = tmp.name

subprocess.run(
    ["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", str(SAMPLE_RATE), wav_path],
    capture_output=True,
)

model = whisperx.load_model("large-v2", "cpu", language="ja")
audio = whisperx.load_audio(wav_path)
result = model.transcribe(audio, language="ja")
segments = result["segments"]
model_a, metadata = whisperx.load_align_model(language_code="ja", device="cpu")
aligned = whisperx.align(segments, model_a, metadata, audio, "cpu", return_char_alignments=False)
aligned_segments = aligned["segments"]

# 全単語リストを作成（segment 境界にマーカーを付ける）
print("=== 単語間ギャップ一覧（segment境界を含む）===\n")

all_words = []
for seg_idx, seg in enumerate(aligned_segments):
    for w in seg.get("words", []):
        if "start" in w and "end" in w:
            all_words.append({
                "word": w.get("word", ""),
                "start": float(w["start"]),
                "end": float(w["end"]),
                "seg": seg_idx,
            })

gaps = []
for i in range(1, len(all_words)):
    prev = all_words[i-1]
    curr = all_words[i]
    # 前単語の推定終端からのギャップ
    estimated_end = prev["start"] + SENTENCE_END_DURATION
    gap = curr["start"] - estimated_end
    seg_boundary = prev["seg"] != curr["seg"]
    gaps.append({
        "i": i,
        "gap": gap,
        "prev_word": prev["word"],
        "curr_word": curr["word"],
        "curr_start": curr["start"],
        "seg_boundary": seg_boundary,
    })

# ギャップ閾値ごとにシミュレーション
print("=== 閾値ごとのキュー数シミュレーション ===\n")
for threshold in [0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]:
    # segment境界 OR ギャップ>閾値 で分割
    splits = [g for g in gaps if g["gap"] > threshold or g["seg_boundary"]]
    n_cues = len(splits) + 1
    print(f"  閾値 {threshold:.1f}s: {n_cues} キュー")

print()

# 閾値1.0秒でのシミュレーション詳細
THRESHOLD = 1.0
MAX_CHARS = 25  # 文字数上限

print(f"=== 閾値 {THRESHOLD}s + 最大{MAX_CHARS}文字でのキュー分割シミュレーション ===\n")

buffer_words = []
cues_sim = []

def flush(buf):
    if not buf:
        return
    text = "".join(w["word"] for w in buf)
    start = buf[0]["start"]
    end = buf[-1]["start"] + SENTENCE_END_DURATION
    cues_sim.append({"start": start, "end": end, "text": text, "n": len(buf)})

for i, w in enumerate(all_words):
    if not buffer_words:
        buffer_words.append(w)
        continue
    prev = buffer_words[-1]
    estimated_end = prev["start"] + SENTENCE_END_DURATION
    gap = w["start"] - estimated_end
    seg_changed = prev["seg"] != w["seg"]
    text_so_far = "".join(x["word"] for x in buffer_words)
    # 分割条件: ギャップ超過 OR segment境界 OR 文字数超過
    if gap > THRESHOLD or seg_changed or len(text_so_far) >= MAX_CHARS:
        flush(buffer_words)
        buffer_words = [w]
    else:
        buffer_words.append(w)

flush(buffer_words)

for i, c in enumerate(cues_sim):
    n_chars = len(c["text"])
    print(f"[{i:3d}] {fmt(c['start'])} --> {fmt(c['end'])}  ({n_chars}文字)")
    print(f"      「{c['text']}」")

Path(wav_path).unlink(missing_ok=True)
