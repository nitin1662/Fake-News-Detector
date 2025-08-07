from flask import Flask, request, jsonify, render_template
import joblib
import re
import string

# Load model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text preprocessing
def wordopt(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"RT|cc", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"[0-9]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        news_text = data.get("text", "")

        processed_text = wordopt(news_text)
        vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)

        label = "Fake News" if prediction == 0 else "Not a Fake News"
        confidence = round(max(proba[0]) * 100, 2)

        return jsonify({
            "prediction": label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
