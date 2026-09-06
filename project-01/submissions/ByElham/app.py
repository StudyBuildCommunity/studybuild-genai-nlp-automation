"""
Streamlit Demo for the AI Customer Support Ticket Classifier.

Usage:
    streamlit run app.py

The app works with no prior setup: on first launch it trains and saves the
model automatically (a few minutes), and every later launch loads the saved
artifacts instantly.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import load_or_train_model, predict_intent

st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="centered",
)


@st.cache_resource
def get_model_and_metadata():
    """Load saved model + metadata, or train-and-save them on first launch."""
    return load_or_train_model()


model, metadata = get_model_and_metadata()

st.title("AI Customer Support Ticket Classifier")
st.markdown(
    "Enter a customer support message. The model predicts the intent "
    "and indicates whether human review is recommended."
)

user_input = st.text_area(
    "Customer message",
    placeholder="e.g., My card was stolen yesterday, please block it.",
    height=120,
)

if st.button("Classify", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message to classify.")
    else:
        result = predict_intent(user_input, model=model, metadata=metadata)

        st.subheader("Prediction")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Intent", result["predicted_intent"])
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1%}")

        if result["needs_human_review"]:
            st.warning(
                f"**Human Review Recommended** — {result['review_reason']}"
            )
        else:
            st.success("Confidence above threshold. Auto-routing is safe.")

        st.subheader("Class Probabilities")
        if result["probabilities"]:
            sorted_probs = sorted(
                result["probabilities"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for intent, prob in sorted_probs:
                st.progress(prob, text=f"{intent}: {prob:.1%}")

st.markdown("---")
st.caption(
    "Model: TF-IDF + Logistic Regression | "
    f"Threshold: {metadata.get('confidence_threshold', 0.5)} | "
    "Dataset: BANKING77 (CC BY 4.0)"
)
