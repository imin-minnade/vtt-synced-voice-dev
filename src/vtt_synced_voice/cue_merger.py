from __future__ import annotations

from .vtt_io import VttCue


def merge_cues(cues: list[VttCue], language: str) -> list[VttCue]:
    """過分割されたVttCueを文単位にマージして返す。

    言語ごとに文末検出ロジックを切り替える:
    - ja: Janome 形態素解析で末尾品詞を判定
    - その他: ピリオド/感嘆符/疑問符で判定
    """
    if not cues:
        return []

    if language == "ja":
        is_end = _make_ja_detector()
    else:
        is_end = _is_end_punctuation

    merged: list[VttCue] = []
    buffer: list[VttCue] = []

    for cue in cues:
        buffer.append(cue)
        merged_text = "".join(c.text for c in buffer)
        if is_end(merged_text):
            merged.append(_flush(buffer, len(merged)))
            buffer = []

    if buffer:
        merged.append(_flush(buffer, len(merged)))

    return merged


def _flush(buffer: list[VttCue], index: int) -> VttCue:
    text = "".join(c.text for c in buffer)
    return VttCue(
        index=index,
        start=buffer[0].start,
        end=buffer[-1].end,
        text=text,
        original_start=buffer[0].original_start,
        original_end=buffer[-1].original_end,
    )


def _make_ja_detector():
    """Janome を使った日本語文末判定クロージャを返す。"""
    from janome.tokenizer import Tokenizer
    tokenizer = Tokenizer()

    def is_end(text: str) -> bool:
        tokens = list(tokenizer.tokenize(text))
        if not tokens:
            return False
        last = tokens[-1]
        pos = last.part_of_speech.split(",")
        pos0 = pos[0]
        pos1 = pos[1] if len(pos) > 1 else "*"

        # 助動詞（です/ます/ました/ません 等）
        # 「た」単体は連体修飾（「作成された」「使った」）と文末が区別できないため
        # 直前トークンが「まし」なら文末の「ました」、それ以外は連体修飾とみなしスキップ
        if pos0 == "助動詞":
            if last.surface == "た":
                prev_surface = tokens[-2].surface if len(tokens) >= 2 else ""
                return prev_surface == "まし"
            return True

        # 動詞・非自立（ください/てごらん 等）
        if pos0 == "動詞" and pos1 == "非自立":
            return True

        # 感動詞（ああ/はい 等）
        if pos0 == "感動詞":
            return True

        return False

    return is_end


def _is_end_punctuation(text: str) -> bool:
    """英語等：ピリオド/感嘆符/疑問符で文末判定。"""
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in ".!?"
