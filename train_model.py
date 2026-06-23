import pandas as pd
import re
import emoji
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Clean text
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


print("Loading Dataset...")

df = pd.read_csv("dataset/IMDB Dataset.csv")


print(f"Dataset Size: {len(df)}")
print(df.head())

print("Cleaning text...")
df["review"] = df["review"].apply(clean_text)

print("Text cleaning completed.")

X = df["review"]
y = df["sentiment"]

print("Starting TF-IDF...")

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1,3),
    min_df=2
)

X = vectorizer.fit_transform(X)

print("TF-IDF completed.")
print("Splitting dataset...")

print("Starting model training...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Naive Bayes model...")

model = MultinomialNB()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(classification_report(y_test, predictions))

print("Training completed.")
print(f"Accuracy: {accuracy:.4f}")

joblib.dump(
    model,
    "model/sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "model/vectorizer.pkl"
)


print("Model Saved Successfully!")