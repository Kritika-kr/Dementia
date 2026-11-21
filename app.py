import streamlit as st

from src.asr import ASR_MODEL, MAX_AUDIO_SECONDS, load_model, transcribe_bytes
from src.predict_single import predict

MIN_WORDS_WARN = 20   # training transcripts are typically 30+ words

st.set_page_config(page_title="Dementia Detection (Voting Ensemble)", layout="centered")
st.title("Dementia Detection App")
st.subheader("Text-based classification using a Voting Ensemble on lexical and embedding features")
st.markdown(
    "This app classifies a speech transcript using an ensemble of "
    "L1-regularized Logistic Regression, Random Forest, and SVM, trained on "
    "TF-IDF, word-count, Word2Vec, GloVe, and spaCy embedding features. "
    "You can type or paste a transcript, or record or upload audio and the app will "
    "transcribe it first."
)


@st.cache_resource(show_spinner="Loading the speech recognition model (first use only)...")
def get_asr():
    return load_model(ASR_MODEL)


if "transcript" not in st.session_state:
    st.session_state["transcript"] = ""
if "from_asr" not in st.session_state:
    st.session_state["from_asr"] = False


def _mark_edited():
    # Called only when the user edits the box, not when we fill it from audio.
    st.session_state["from_asr"] = False


# ---------------------------------------------------------------- audio input
st.markdown("#### Option 1: speak or upload audio")
audio_bytes = None
if hasattr(st, "audio_input"):   # needs streamlit >= 1.39
    recording = st.audio_input("Record your description of the picture")
    if recording is not None:
        audio_bytes = recording.getvalue()
uploaded = st.file_uploader(
    "Or upload an audio file (an uploaded file is used instead of a recording)",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
)
if uploaded is not None:
    audio_bytes = uploaded.getvalue()

st.caption(
    f"English only, up to {MAX_AUDIO_SECONDS} seconds. Audio is processed in memory to "
    "produce the transcript and is not saved by this app."
)

if st.button("Transcribe audio", disabled=audio_bytes is None):
    with st.spinner("Transcribing..."):
        try:
            text = transcribe_bytes(get_asr(), audio_bytes)
        except Exception as exc:   # bad file, too long, model download failed, ...
            st.error(f"Could not transcribe the audio: {exc}")
        else:
            if text:
                st.session_state["transcript"] = text
                st.session_state["from_asr"] = True
            else:
                st.warning("No speech was detected in the audio.")

# ------------------------------------------------------------ transcript box
st.markdown("#### Option 2: type, paste, or correct the transcript")
st.text_area(
    "Speech transcript (e.g., a patient's response to a picture-description task):",
    key="transcript",
    height=200,
    on_change=_mark_edited,
)
if st.session_state["from_asr"]:
    st.caption(
        "Transcribed automatically. Please read it and correct any mistakes before "
        "predicting. The model was trained on manually transcribed text that includes "
        "hesitations such as 'um' and 'uh'; automatic transcripts often leave these out, "
        "so predictions from audio are less reliable than from a careful transcript."
    )

# ---------------------------------------------------------------- prediction
if st.button("Predict", type="primary"):
    text = st.session_state["transcript"].strip()
    if not text:
        st.warning("Please enter or record something first.")
    else:
        if len(text.split()) < MIN_WORDS_WARN:
            st.warning(
                f"This transcript has fewer than {MIN_WORDS_WARN} words. Very short "
                "inputs give unreliable predictions."
            )
        with st.spinner("Extracting features and predicting..."):
            result = predict(text)
        st.success(f"**Prediction:** {result['label']}")
        conf = result["confidence"]
        st.info(
            f"**Confidence:** Control: {conf['Control']:.2f}, "
            f"Dementia: {conf['Dementia']:.2f}"
        )
        if st.session_state["from_asr"]:
            st.caption("This prediction is based on an automatic transcript that you did not edit.")
        st.caption(
            "This tool is a research prototype and is not a diagnostic device. "
            "Consult a qualified clinician for any medical concern."
        )