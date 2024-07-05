import os
import sys
import time
from typing import List, Optional, Union

from threading import Lock
import numpy as np

from leopardai.photon import Photon, HTTPException
from leopardai.photon.types import File

from loguru import logger


class WhisperX(Photon):
    """
    A WhisperX photon that serves the [WhisperX](https://github.com/m-bain/whisperX) model.

    The photon exposes a single endpoint "/run" that takes an audio file as input, and returns
    the transcription, and alignment, and diarization results.
    """

    # Note: openai-whisper implicitly requires triton 2.1.0, which in turn might be in conflict
    # with non-version-pinned torch and torchaudio. As a result, we will pin all three versions
    # here.
    requirement_dependency = [
        "numpy",
        "torchaudio",
        "pyannote.audio==3.1.1",
        "speechbrain==0.5.14",  # to overcome recent bug in speechbrain 1.0.0
        "git+https://github.com/m-bain/whisperx.git@78dcfa",
    ]

    system_dependencies = ["ffmpeg"]

    deployment_template = {
        "resource_shape": "gpu.a10",
        "env": {
            "WHISPER_MODEL": "large-v3",
            # maximum audio length that the api allows. In default, we will use
            # 10 minutes. If you are deploying things on your own, you can change
            # it to be longer.
            "MAX_LENGTH_IN_SECONDS": "600",
        },
        "secret": [
            "HUGGING_FACE_HUB_TOKEN",
        ],
    }

    # If one is doing a lot of alignments and diarizations, it is possible that
    # the gpu is underutilized. In this case, one can increase the concurrency
    # to better utilize the gpu.
    handler_max_concurrency = 8

    SUPPORTED_LANGUAGES = {"en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt"}
    # The main language for the model
    MAIN_LANGUAGE = "en"
    # batch size that is benchmarked to be the best balance on A10
    DEFAULT_BATCH_SIZE = 16

    # Because each alignment language takes a nontrivial amount of memory,
    # we only keep languages that we find are commonly called, and load other
    # models on-demand. You can change this to host more alignment models in a
    # warm state at the cost of more memory.
    ALIGNMENT_LANGUAGE_TO_KEEP = {"en", "zh", "es"}

    def init(self):
        import torch
        import whisperx
        from whisperx.asr import FasterWhisperPipeline

        # This is a temporary workaround on our platform to work around the cuda
        # 11 issue. If you are running things locally, you can install cuda 11
        # instead of cuda 12, due to faster-whisper and pyannote.audio's dependency.
        # Of course, this is flaky - but so far ctranslate2 have been sticking with
        # standard cuda apis and does not cause much issue.
        if (
            "leopard_WORKSPACE_ID" in os.environ
            and "leopard_DEPLOYMENT_NAME" in os.environ
        ):
            os.system(
                "ln -s /usr/local/cuda/lib64/libcublas.so.12"
                " /usr/local/cuda/lib64/libcublas.so.11"
            )

        logger.info("Initializing WhisperX")

        self.USE_FASTER_WHISPER = True
        self.WHISPER_MODEL = os.environ["WHISPER_MODEL"]
        self.MAX_LENGTH_IN_SECONDS = int(os.environ["MAX_LENGTH_IN_SECONDS"])

        # 1. Load whisper model
        self.hf_token = os.environ["HUGGING_FACE_HUB_TOKEN"]
        if not self.hf_token:
            logger.error("Please set the environment variable HUGGING_FACE_HUB_TOKEN.")
            sys.exit(1)
        if torch.cuda.is_available():
            self.device = "cuda"
            compute_type = "float16"
        else:
            self.device = "cpu"
            compute_type = "float32"

        # 1. load whisper model
        # We keep a main model as MAIN_LANGUAGE so we don't need to always reload
        # tokenizers. We also keep a multilingual model that can handle all languages.
        self._main_model = whisperx.load_model(
            self.WHISPER_MODEL,
            self.device,
            compute_type=compute_type,
            language=self.MAIN_LANGUAGE,
        )
        self._multilingual_model = FasterWhisperPipeline(
            model=self._main_model.model,
            vad=self._main_model.vad_model,
            options=self._main_model.options,
            tokenizer=None,
            language=None,
            suppress_numerals=self._main_model.suppress_numerals,
            vad_params=self._main_model._vad_params,
        )
        # For the main model, inference is not thread safe (because of some underlying cuda memory
        # accesses). As a result, whenever we use the transcribe model, we need to lock it.
        self.transcribe_model_lock = Lock()

        # 2. load whisper align model. Alignment models are language specific, so we will basically
        # load them as a dictionary. In addition, we only load models that are in ALIGNMENT_LANGUAGE_TO_KEEP.
        self._model_a = {}
        self._metadata = {}
        for lang in self.ALIGNMENT_LANGUAGE_TO_KEEP:
            self._model_a[lang], self._metadata[lang] = whisperx.load_align_model(
                language_code=lang, device=self.device
            )
        # Since we don't know if whisper's align function is perfectly thread safe or not, we
        # will lock it as well.
        self.align_model_lock = Lock()

        # 3. load whisper diarize model. Diarization model right now is thread safe.
        self._diarize_model = whisperx.DiarizationPipeline(
            model_name="pyannote/speaker-diarization@2.1",
            use_auth_token=self.hf_token,
            device=self.device,
        )
        self._diarize_model_lock = Lock()

    def _transcribe(self, audio: np.ndarray, language: Optional[str] = None):
        logger.debug("transcribe: aquiring lock")
        with self.transcribe_model_lock:
            logger.debug("transcribe: lock acquired")
            if language == self.MAIN_LANGUAGE:
                result = self._main_model.transcribe(
                    audio, batch_size=self.DEFAULT_BATCH_SIZE, language=language
                )
            else:
                result = self._multilingual_model.transcribe(
                    audio, batch_size=self.DEFAULT_BATCH_SIZE, language=language
                )
        logger.debug("transcribe: lock released")
        return result

    def _align(self, result, audio):
        # Run alignment
        import whisperx

        logger.debug("Start alignment")
        if result["language"] in self.SUPPORTED_LANGUAGES:
            model_a = self._model_a[result["language"]]
            metadata_a = self._metadata[result["language"]]
        else:
            # load model_a and metadata on-demand
            model_a, metadata_a = whisperx.load_align_model(
                language_code=result["language"], device=self.device
            )
        with self.align_model_lock:
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata_a,
                audio,
                self.device,
                return_char_alignments=False,
            )
        logger.debug("alignment done.")
        return result