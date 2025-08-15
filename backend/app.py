from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import os


# Define paths relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.abspath(os.path.join(script_dir, '..', 'frontend'))
model_path = os.path.join(script_dir, 'model', 'spam_classifier.pkl')

# Check for model file before initializing the app further
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"The model file was not found at {os.path.abspath(model_path)}. "
        "Please run 'python backend/train_model.py' first to generate it."
    )

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the model and vectorizer
with open(model_path, 'rb') as f:
    vectorizer, model = pickle.load(f)

# --- Static file serving for the frontend ---
@app.route('/')
def index():
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory(frontend_dir, 'styles.css')

# --- API endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input: No JSON data received."}), 400

        email_id = data.get('email_id', '')
        subject = data.get('subject', '')
        message = data.get('message', '')  # Frontend sends 'message'

        full_text = f"{email_id} {subject} {message}"
        vector = vectorizer.transform([full_text])
        prediction = model.predict(vector)[0]

        return jsonify({"result": "SPAM" if prediction == 1 else "NOT SPAM"})
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
