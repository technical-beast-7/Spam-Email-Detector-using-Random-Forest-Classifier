# Spam Email Detector using Random Forest Classifier

This project is a full-stack web application that uses a **Random Forest** machine learning model to classify emails as "Spam" or "Ham" (not spam). It features a clean frontend interface that communicates with a Flask backend API to deliver real-time predictions.

## ✨ Features

- **Accurate ML Model**: Utilizes a Random Forest classifier trained on a custom dataset, achieving over 93% accuracy.
- **Interactive Frontend**: A user-friendly interface built with HTML, CSS, and JavaScript for submitting emails.
- **RESTful API**: A robust backend built with Flask to serve the model and handle prediction requests.
- **Production-Ready**: Configured for deployment on cloud platforms like Render using Gunicorn.
- **Automated Training**: The model is automatically trained during the deployment build process.

## 💻 Technology Stack

- **Backend**: Python, Flask, Gunicorn
- **Machine Learning**: Scikit-learn, Pandas
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Render

## 📂 Project Structure

```
Email_Spam_Detection-main/
├── backend/
│   ├── dataset/
│   │   └── email_dataset_full_with_email.csv
│   ├── model/
│   │   └── spam_classifier.pkl
│   ├── app.py                  # Flask API server
│   ├── render.yaml             # Deployment configuration for Render
│   ├── requirements.txt        # Python dependencies
│   └── train_model.py          # Script to train the ML model
├── frontend/
│   ├── index.html              # Main web page
│   └── styles.css              # Stylesheet
└── README.md
```

## 🚀 Local Setup and Installation

Follow these steps to run the project on your local machine.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/technical-beast-7/Spam-Email-Detector-using-Random-Forest-Classifier.git
    cd Spam-Email-Detector-using-Random-Forest-Classifier
    ```

2.  **Install Dependencies**
    It's recommended to use a virtual environment.
    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install required packages
    pip install -r backend/requirements.txt
    ```

3.  **Train the Model**
    This script will process the dataset and save the trained model (`spam_classifier.pkl`) in the `backend/model/` directory.
    ```bash
    python backend/train_model.py
    ```

4.  **Run the Application**
    This command starts the Flask development server.
    ```bash
    python backend/app.py
    ```

5.  **Access the Web App**
    Open your web browser and navigate to `http://127.0.0.1:5000`.

## 🤖 API Endpoint

The backend provides a single API endpoint for predictions.

### Predict Spam

- **URL**: `/predict`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "email_id": "sender@example.com",
    "subject": "Your Subject Here",
    "message": "The body of the email message..."
  }
  ```
- **Success Response (200)**:
  ```json
  {
    "result": "SPAM"
  }
  ```
  or
  ```json
  {
    "result": "NOT SPAM"
  }
  ```

## 📈 Model Performance

The Random Forest model was evaluated on a 20% test split of the dataset.

- **📊 Accuracy**: **93.18%**
- **📋 Classification Report**:
  ```
                precision    recall  f1-score   support

           Ham       0.95      0.91      0.93        23
          Spam       0.91      0.95      0.93        21

      accuracy                           0.93        44
     macro avg       0.93      0.93      0.93        44
  weighted avg       0.93      0.93      0.93        44
  ```

## ☁️ Deployment

This project is configured for easy deployment on **Render** using the `backend/render.yaml` file.

- **Build Command**: `pip install -r requirements.txt && python train_model.py`
  - This command first installs all dependencies and then runs the training script to ensure the `spam_classifier.pkl` model file is available before the server starts.
- **Start Command**: `gunicorn app:app`
  - This uses **Gunicorn**, a production-grade WSGI server, to run the Flask application, which is more robust and performant than the built-in development server.
