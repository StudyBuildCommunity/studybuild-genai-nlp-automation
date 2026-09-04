import joblib
from pathlib import Path


# Load the trained model
MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"
model = joblib.load(MODEL_PATH)


# Confidence threshold for human review
CONFIDENCE_THRESHOLD = 0.80


def predict_intent(message):
    """
    Predict the intent of a customer-support message.

    Returns:
        predicted_intent: Predicted category
        confidence: Maximum predicted probability
        decision: Automatic classification or human review
    """

    predicted_intent = model.predict([message])[0]

    probabilities = model.predict_proba([message])[0]
    confidence = probabilities.max()

    if confidence < CONFIDENCE_THRESHOLD:
        decision = "Human review required"
    else:
        decision = "Automatic classification"

    return predicted_intent, confidence, decision


if __name__ == "__main__":
    message = input("Enter a customer-support message: ")

    intent, confidence, decision = predict_intent(message)

    print(f"\nPredicted intent: {intent}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Decision: {decision}")