from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
model_path = os.path.join("notebooks", "models", "sentiment_model.pkl")
vec_path = os.path.join("notebooks", "models", "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vec_path)

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    if request.method == "POST":
        text = request.form.get("text", "")
        if text.strip():
            vec = vectorizer.transform([text])
            pred = model.predict(vec)[0]
            sentiment = "positive" if pred == 1 else "negative"
        else:
            sentiment = "Please enter some text."
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    # Bind to 0.0.0.0 and dynamic port for Render
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
