from typing import Dict, Any, List


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
