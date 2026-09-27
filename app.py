import streamlit as st
from src.predict_single import predict

st.set_page_config(page_title="Dementia Detection (Voting Ensemble)", layout="centered")
st.title("Dementia Detection App")
st.subheader("Text-based classification using a Voting Ensemble on combined linguistic features")
st.markdown(
    "This app classifies a speech transcript using an ensemble of "
    "L1-regularized Logistic Regression, Random Forest, and SVM, trained on "
    "lexical, embedding-based, and handcrafted linguistic features."
)

user_input = st.text_area(
    "Enter speech transcript (e.g., patient's response to a picture-description task):",
    height=200,
)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text input.")
    else:
        with st.spinner("Extracting features and predicting..."):
            result = predict(user_input)
        st.success(f"**Prediction:** {result['label']}")
        conf = result["confidence"]
        st.info(
            f"**Confidence:** Control: {conf['Control']:.2f}, "
            f"Dementia: {conf['Dementia']:.2f}"
        )
        st.caption(
            "This tool is a research prototype and is not a diagnostic device. "
            "Consult a qualified clinician for any medical concern."
        )
