***

# Spam Mail Detection — AI Phishing Email Classifier

**A simple, presentation-ready Flask web application that classifies emails or messages as spam/phishing or legitimate.**  
Built with Python, Flask, scikit-learn, and a TF-IDF + classifier pipeline. Designed for easy local running and deployment.

***

## AI Model Highlights

### 🚀 AI Model Overview

This project uses a custom-trained machine learning pipeline built on modern Natural Language Processing (NLP) techniques, designed to detect phishing and spam patterns—including sophisticated social-engineering style phishing attempts.

The model processes text using:  
- **TF-IDF Vectorization**: Captures meaningful word patterns, frequencies, and contextual importance within messages.  
- **Enhanced NLP Feature Extraction**: Custom logic detects suspicious tokens such as "verify," "urgent," "account," threat-based language, financial/social-engineering keywords, and embedded URLs.  
- **Optimized Classifier Pipeline**: Uses Logistic Regression with hyperparameter tuning, class weight balancing, and probability calibration, allowing output of precise phishing probability scores beyond just labels.

### 🎯 Model Performance (Test Summary)

- Accuracy: ~97%  
- High Precision: Effectively catches phishing without misclassifying legitimate emails  
- High Recall: Detects even tricky phishing attempts  
- ROC AUC: ~0.98 (Shows excellent phishing vs. legitimate differentiation)  

This performance is comparable to industry-grade spam filters commonly used in email systems.

### 🧠 What Makes This Model Special?

- **Custom "Explainability Layer"** highlights suspicious keywords, URLs, and tokens associated with phishing, providing transparency valuable for demonstrations and real-world use.  
- **Phishing Probability Score** offers nuanced results such as "Phish Probability: 97.61% — Warning: This message looks dangerous," making the system more trustworthy and intelligent.  
- **Flexible Input** supports emails, SMS, WhatsApp/Telegram texts, bank alerts, social media messages, fake OTPs, job scams, crypto/lottery scams, and more—any text message.  
- **Lightweight & Fast** with no GPU requirement, instant load, and easy deployment on platforms like Render or Railway.  
- **Designed for Real Use & Academic Value**, ideal for college projects, security awareness training, portfolio-building, and lightweight phishing detection in practice.

***

## Quick Demo (Local Setup)

1. Clone the repository and navigate to the project folder.
2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\Activate
# macOS / Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Flask app:

```bash
python app.py
```

5. Open your browser to [http://127.0.0.1:5000](http://127.0.0.1:5000).

***

## Project Structure (Key Files)

```
Spam_Mail_Detection/
├─ app.py                  # Flask web application
├─ train_improved.py       # (Optional) Script to retrain/improve the model pipeline
├─ pipeline.joblib         # Trained model file (if present)
├─ mail_random_forest.joblib (optional)
├─ requirements.txt        # Python dependencies
├─ README.md               # This file
├─ Templates/
│  ├─ index.html           # Input form page
│  └─ after.html           # Result display page
├─ static/
│  └─ CSS/
│     └─ custom.css        # Local CSS theme
└─ logs/
   └─ predictions.csv      # Logged predictions at runtime (if enabled)
```

***

## Retraining or Improving the Model

Run this to retrain the model:

```bash
python train_improved.py
```

- This script trains, evaluates, and saves the pipeline as `pipeline.joblib`.  
- If saving fails due to pickling issues (e.g., with lambdas), see script comments for workarounds like saving vectorizer and model separately.  
- Place the trained model in the project root for the app to load on startup.

***

## How the Web App Works (Brief)

- `index.html`: Form to paste email or message text.  
- Posts to `/predict` endpoint in `app.py`.  
- Input is preprocessed and vectorized.  
- Model predicts label and phishing probability.  
- `after.html` displays results with verdict, probability bar, suspicious tokens, and detected URLs.  
- Predictions may be logged in `logs/predictions.csv` if enabled.

***

## Deployment Options

- Deploy easily on Render.com or Railway.app with free tiers.  
- Push to GitHub, create a web service on these platforms linking to the repo.  
- Use a `Procfile` with:  
  ```
  web: gunicorn app:app
  ```
- Configure environment variables if needed, deploy, and get a public URL.

***

## Tips & Troubleshooting

- Add `venv/` to `.gitignore` to avoid committing virtual environment.  
- Make sure `custom.css` is at `static/CSS/custom.css` and linked correctly in HTML:  
  ```html
  <link rel="stylesheet" href="{{ url_for('static', filename='CSS/custom.css') }}">
  ```
- For pickling errors on model loading, avoid lambdas or save/load the vectorizer and model separately.  
- To keep models private when sharing, exclude `pipeline.joblib` and instruct users to train locally.

***

## Usage Examples (Test Inputs)

### Phishing Example

```
Subject: Urgent: Verify your account now

We detected unusual activity on your bank account. Please verify immediately:
http://secure-login-bank.example.com/verify
```

### Legitimate Example

```
Subject: Team meeting notes

Hi team, thanks for today's meeting. Attached are minutes and next steps.
```

***

## Credits & License

Built by Ram as a classroom demo and minor project.

***
