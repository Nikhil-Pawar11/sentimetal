from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Define the path to the model and vectorizer files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sentiment_analysis_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

# Load the sentiment analysis model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return prediction[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('queryResult', {}).get('queryText', '')
    sentiment = predict_sentiment(text)
    return jsonify({'fulfillmentText': f'The sentiment is {sentiment}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
