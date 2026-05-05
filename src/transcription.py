import gc
import logging
import os
from pathlib import Path

import pandas as pd
import torch
import torchaudio
import whisperx
from dotenv import load_dotenv
from pyannote.audio import Pipeline

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

        diarization_cfg = config.get("diarization", {}) or {}
        self.num_speakers = diarization_cfg.get("num_speakers")
        self.min_speakers = diarization_cfg.get("min_speakers")
        self.max_speakers = diarization_cfg.get("max_speakers")

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

    def diarize(self, waveform, sample_rate, aligned):
        """
        waveform: torch.Tensor shape (channels, time)
        sample_rate: int
        aligned: результат whisperx.align
        """
        try:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                log.warning("HF_TOKEN не найден — пропускаем диаризацию")
                return aligned

            if self.diarize_model is None:
                log.info("Loading diarization pipeline...")
                self.diarize_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token,
                )
                self.diarize_model.to(torch.device(self.device))

            audio_input = {
                "waveform": waveform,  # torch.Tensor (channels, time)
                "sample_rate": sample_rate  # int, обычно 16000
            }

            log.info("Running speaker diarization...")
            diarization_kwargs = {}
            if self.num_speakers is not None:
                diarization_kwargs["num_speakers"] = int(self.num_speakers)
            else:
                if self.min_speakers is not None:
                    diarization_kwargs["min_speakers"] = int(self.min_speakers)
                if self.max_speakers is not None:
                    diarization_kwargs["max_speakers"] = int(self.max_speakers)

            diarization = self.diarize_model(audio_input, **diarization_kwargs)

            # pyannote может вернуть либо Annotation, либо DiarizeOutput (с serialize()).
            if hasattr(diarization, "serialize"):
                serialized = diarization.serialize()
                # Для assign_word_speakers лучше использовать "exclusive" разметку без перекрытий.
                diarize_segments = pd.DataFrame(
                    serialized.get("exclusive_diarization") or serialized.get("diarization", [])
                )
            else:
                annotation = (
                    getattr(diarization, "exclusive_speaker_diarization", None)
                    or getattr(diarization, "speaker_diarization", None)
                    or diarization
                )
                diarize_segments_list: list[dict] = []
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    diarize_segments_list.append(
                        {
                            "start": float(turn.start),
                            "end": float(turn.end),
                            "speaker": str(speaker),
                        }
                    )
                diarize_segments = pd.DataFrame(diarize_segments_list)

            required_columns = {"start", "end", "speaker"}
            if diarize_segments.empty or not required_columns.issubset(diarize_segments.columns):
                log.warning("Diarization result has unexpected format; returning aligned transcript.")
                return aligned

            return whisperx.assign_word_speakers(
                diarize_segments,
                aligned,
                fill_nearest=False
            )

        except Exception as e:
            log.error(f"Diarization failed: {e}")
            return aligned

    @staticmethod
    def preprocess_audio(path: str, sample_rate: int = 16000):
        """Загружает и нормализует аудио, возвращает tensor + sample_rate."""
        waveform, sr = torchaudio.load(path)

        # Конвертируем в mono если стерео
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

        # Нормализация
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        return waveform, sample_rate

    def transcribe(self, audio_path: str):
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        log.info(f"Preprocessing audio: {audio_path}")
        waveform, sample_rate = self.preprocess_audio(str(audio_path))

        self.load_whisper()

        log.info("Transcribing...")
        audio_np = waveform.squeeze().numpy()

        result = self.whisper_model.transcribe(
            audio_np,
            batch_size=self.batch_size,
            language=self.language
        )

        return waveform, sample_rate, result

    def align(self, waveform, sample_rate, result):
        self.load_alignment(result["language"])

        log.info("Aligning...")
        return whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            waveform,  # torch.Tensor
            self.device,
            return_char_alignments=False
        )

    @staticmethod
    def free_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, audio_path: str):
        # Получаем waveform, sample_rate и результат транскрибации
        waveform, sample_rate, result = self.transcribe(audio_path)

        # Выравнивание
        aligned = self.align(waveform, sample_rate, result)

        # Освобождаем память от Whisper перед diarization
        del self.whisper_model
        self.whisper_model = None
        self.free_memory()

        final = self.diarize(waveform, sample_rate, aligned)

        return final
