import gc
import logging
import os
from pathlib import Path
from pyannote.audio import Pipeline
import numpy as np
import torch
import whisperx
from dotenv import load_dotenv

from src.config import Config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s"
)
log = logging.getLogger(__name__)

config = Config()


class TranscriptionPipeline:
    def __init__(self, whisper_model: str | None = None, device: str | None = None):
        self.device = device or config["device"] or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = whisper_model or config["models"]["whisper"]
        self.batch_size = config["batch_size"]
        self.language = config["language"]

        self.compute_type = config["compute_type"]
        if self.device == "cpu" and self.compute_type not in ["int8", "int8_float32"]:
            self.compute_type = "int8"

        self.whisper_model = None
        self.diarize_model = None
        self.align_model = None
        self.align_metadata = None

    def load_whisper(self):
        if self.whisper_model is None:
            log.info("Loading Whisper model...")
            self.whisper_model = whisperx.load_model(
                self.model_name,
                self.device,
                language=self.language,
                compute_type=self.compute_type
            )

    def load_alignment(self, language_code: str):
        """Загрузка модели выравнивания (alignment)."""
        if self.align_model is None:
            log.info(f"Loading alignment model for: {language_code}...")
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device
            )

    def diarize(self, audio, aligned):
        try:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                log.warning("HF_TOKEN не найден")
                return aligned

            if not hasattr(self, "diarize_model") or self.diarize_model is None:
                self.diarize_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token
                )

            diarization = self.diarize_model(audio)

            diarize_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                diarize_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })

            return whisperx.assign_word_speakers(
                diarize_segments,
                aligned,
                fill_nearest=True
            )

        except Exception as e:
            log.error(f"Diarization failed: {e}")
            return aligned

    @staticmethod
    def preprocess_audio(path: str):
        audio = whisperx.load_audio(path)

        # нормализация
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        return audio

    def transcribe(self, audio_path: str):
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        log.info(f"Preprocessing audio: {audio_path}")
        audio = self.preprocess_audio(str(audio_path))

        self.load_whisper()

        log.info("Transcribing...")
        result = self.whisper_model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=self.language
        )

        return audio, result

    def align(self, audio, result):
        self.load_alignment(result["language"])

        log.info("Aligning...")
        return whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            self.device,
            return_char_alignments=False
        )

    @staticmethod
    def free_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, audio_path: str):
        audio, result = self.transcribe(audio_path)
        aligned = self.align(audio, result)

        del self.whisper_model
        self.whisper_model = None
        self.free_memory()

        final = self.diarize(audio, aligned)
        return final
