import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import librosa
import noisereduce as nr
import soundfile as sf


def save_transcript_with_summary(
    segments: List[Dict[str, Any]],
    summary: str,
    output_file: str
):
    with open(output_file, "w", encoding="utf-8") as f:
        # Сначала суммаризация
        f.write("=== SUMMARY ===\n")
        f.write(summary + "\n\n")
        # Потом сам текст
        f.write("=== FULL TRANSCRIPT ===\n")
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg["text"]
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            f.write(f"{speaker} [{start:.1f}-{end:.1f}]: {text}\n")


def process_media(input_path: str, output_path: str, reduce_noise: bool = False):
    """
    Извлекает аудио (если это видео), конвертирует в 16kHz mono WAV
    и опционально применяет шумоподавление.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        temp_wav = tmp_file.name

    command = [
        "ffmpeg", "-y", "-i", str(input_file),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", temp_wav
    ]
    try:
        ffmpeg_proc = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        raise RuntimeError(
            "FFmpeg не найден в PATH. Установите FFmpeg и проверьте команду `ffmpeg -version`."
        ) from exc
    if ffmpeg_proc.returncode != 0:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        stderr = ffmpeg_proc.stderr.strip() or "Unknown ffmpeg error"
        raise RuntimeError(f"Ошибка обработки медиа через ffmpeg: {stderr}")

    # Шумоподавление (если включено)
    if reduce_noise:
        y, sr = librosa.load(temp_wav, sr=16000)
        # Применяем спектральное вычитание шума
        reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
        sf.write(output_path, reduced_noise, sr)
    else:
        os.replace(temp_wav, output_path)

    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    return output_path
