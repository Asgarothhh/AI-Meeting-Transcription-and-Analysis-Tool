import logging
import os
from pathlib import Path
import torch
from config import Config
from transcription import TranscriptionPipeline
from summarization import graph
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
from utils import save_transcript_with_summary

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s"
)
log = logging.getLogger(__name__)

config = Config()

model_dir = Path(config["paths"]["model_dir"])
hf_cache = Path(config["paths"]["hf_cache"])
torch_cache = Path(config["paths"]["torch_cache"])
whisper_cache = Path(config["paths"]["whisper_cache"])

for p in [model_dir, hf_cache, torch_cache, whisper_cache]:
    p.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(hf_cache)
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
os.environ["HF_HUB_CACHE"] = str(hf_cache)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_HOME"] = str(torch_cache)
os.environ["WHISPER_CACHE"] = str(whisper_cache)

DEVICE = config["device"] or ("cuda" if torch.cuda.is_available() else "cpu")

COMPUTE_TYPE = config["compute_type"]
if DEVICE == "cpu" and COMPUTE_TYPE not in ["int8", "int8_float32"]:
    COMPUTE_TYPE = "int8"

BATCH_SIZE = config["batch_size"]
MODEL_NAME = config["models"]["whisper"]
LANGUAGE = config["language"]


if __name__ == "__main__":
    pipeline = TranscriptionPipeline()

    audio_file = "dialogue.wav"
    output_file = "transcript_with_summary.md"

    log.info("Starting full pipeline...")
    result = pipeline.run(audio_file)

    segments = result["segments"]

    transcript_string = "\n".join(
        [f"{s.get('speaker','UNKNOWN')}: {s['text']}" for s in segments]
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(transcript_string)

    log.info("Running LLM summarization...")
    summary_result = graph.invoke({"contents": chunks})
    final_summary = summary_result["final_summary"]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("===== MEETING SUMMARY =====\n\n")
        f.write(final_summary)
        f.write("\n\n===== FULL TRANSCRIPT =====\n\n")

        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg["text"]
            start = seg.get("start", 0)
            end = seg.get("end", 0)

            f.write(f"{speaker} [{start:.1f}-{end:.1f}]: {text}\n")

    log.info(f"Saved → {output_file}")
