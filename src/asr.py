"""
Speech-to-text front end for the Streamlit app.

Audio is transcribed locally with faster-whisper, then the text goes through the
same pipeline as typed transcripts (src.predict_single.predict).

IMPORTANT: the classifier was trained on MANUAL transcripts that keep fillers
("uh", "um"). Whisper normally drops them, so the prompt below nudges it to keep
them. Predictions made from automatic transcripts have NOT been evaluated.
"""
import io
import os

ASR_MODEL = os.environ.get("ASR_MODEL", "base.en")   # use "tiny.en" if memory is tight
MAX_AUDIO_SECONDS = 180

# Neutral prompt containing fillers; it makes Whisper more likely to transcribe
# "um"/"uh" instead of silently removing them. Deliberately has no task content,
# so the model has nothing picture-related to echo back.
FILLER_PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."


def load_model(model_name: str = ASR_MODEL):
    """Load the faster-whisper model (CPU, int8). Call through st.cache_resource."""
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe_bytes(model, data: bytes, max_seconds: int = MAX_AUDIO_SECONDS) -> str:
    """Transcribe audio bytes (wav/mp3/m4a/ogg/flac) to plain English text.

    Raises ValueError if the clip is longer than max_seconds.
    """
    segments, info = model.transcribe(
        io.BytesIO(data),
        language="en",
        beam_size=5,
        vad_filter=False,                  # VAD can strip short fillers
        condition_on_previous_text=False,  # avoids repetition loops on long clips
        initial_prompt=FILLER_PROMPT,
    )
    if info.duration and info.duration > max_seconds:
        raise ValueError(
            f"The audio is {info.duration:.0f} s long; the limit is {max_seconds} s."
        )
    return " ".join(seg.text.strip() for seg in segments).strip()