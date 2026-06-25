from flask import Flask, render_template, request
import joblib
import emoji
import re

app = Flask(__name__)

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


def clean_text(text):

    text = str(text).lower()

    text = emoji.demojize(text)

    # Positive Emojis
    text = text.replace(
        "smiling_face_with_smiling_eyes",
        " positive "
    )

    text = text.replace(
        "grinning_face",
        " positive "
    )

    text = text.replace(
        "red_heart",
        " positive "
    )

    text = text.replace(
        "thumbs_up",
        " positive "
    )

    # Negative Emojis
    text = text.replace(
        "angry_face",
        " negative "
    )

    text = text.replace(
        "thumbs_down",
        " negative "
    )

    text = text.replace(
        "crying_face",
        " negative "
    )

    text = re.sub(r"http\S+", "", text)

    text = re.sub(
        r"[^a-zA-Z\s_]",
        "",
        text
    )

    return text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["text"]

    cleaned_text = clean_text(text)

    vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(vector)[0].lower()

    confidence = round(
        max(model.predict_proba(vector)[0]) * 100,
        2
    )

    # Optional Neutral Logic
    if confidence < 60:
        prediction = "neutral"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        text=text
    )

if __name__ == "__main__":

     app.run(debug=True) 