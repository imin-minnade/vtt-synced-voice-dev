"""ギャップ閾値ごとのキュー分割結果を比較するスクリプト（文字数制限なし）。"""
import subprocess
import tempfile
from pathlib import Path

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

def simulate(threshold: float, use_seg_boundary: bool = True):
    buffer_words = []
    cues = []

    def flush(buf):
        if not buf:
            return
        text = "".join(w["word"] for w in buf)
        start = buf[0]["start"]
        end = buf[-1]["start"] + SENTENCE_END_DURATION
        cues.append({"start": start, "end": end, "text": text})

    for w in all_words:
        if not buffer_words:
            buffer_words.append(w)
            continue
        prev = buffer_words[-1]
        gap = w["start"] - (prev["start"] + SENTENCE_END_DURATION)
        seg_changed = use_seg_boundary and (prev["seg"] != w["seg"])
        if gap > threshold or seg_changed:
            flush(buffer_words)
            buffer_words = [w]
        else:
            buffer_words.append(w)

    flush(buffer_words)
    return cues

for threshold in [1.0, 1.5, 2.0]:
    cues = simulate(threshold)
    print(f"\n{'='*60}")
    print(f"閾値 {threshold}s（segment境界あり）: {len(cues)} キュー")
    print('='*60)
    for i, c in enumerate(cues):
        n = len(c["text"])
        print(f"[{i:2d}] {fmt(c['start'])} --> {fmt(c['end'])}  ({n}文字)  「{c['text']}」")

Path(wav_path).unlink(missing_ok=True)
