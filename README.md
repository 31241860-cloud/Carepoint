# CarePoint — AI Doctor Recommendation & Online Appointment Booking

A college demonstration project for a Smart Hospital Management System. It combines a Flask web application, SQLite appointment storage, and a lightweight NLP text-matching recommender.

## What it does
- Accepts a patient's free-text health concern.
- Uses scikit-learn TF-IDF vectors and cosine similarity to compare the text with example descriptions.
- Recommends a medical specialty and displays fictional sample doctor profiles.
- Saves demo appointment requests in a local SQLite database.
- Displays a doctor directory and saved demo appointments.

## Run on Windows (VS Code)
1. Install Python 3.10 or newer.
2. Extract this ZIP and open the `CarePoint_AI_Hospital` folder in VS Code.
3. Open Terminal in that folder.
4. Create a virtual environment (recommended):
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```
5. Install requirements:
   ```bash
   python -m pip install -r requirements.txt
   ```
6. Start the application:
   ```bash
   python app.py
   ```
7. Open `http://127.0.0.1:5000` in your browser.

If `py` is unavailable, try `python` instead. The `hospital.db` file is created automatically on first run.

## How the AI/NLP component works
1. A small set of sample symptom descriptions is labelled with a specialist type.
2. `TfidfVectorizer` converts the text into numerical word/phrase vectors.
3. `cosine_similarity` compares the patient's text with the example descriptions.
4. The closest example's label becomes the suggested specialty.
5. If the match is weak, the app falls back to General Physician.

This is a small educational prototype, not a clinically validated model. The displayed similarity percentage is a text-match score, not the probability that the specialty is correct. Expand and validate the dataset with qualified medical input before any real-world use.

## Important notes
- All doctor names and credentials in this demo are fictional placeholders.
- No real appointments are made and no real hospital is connected.
- This app does not diagnose, prescribe, or replace a qualified clinician.
- For emergency symptoms, contact local emergency services rather than relying on this app.
- Before deployment, replace the demo secret key, add proper security/validation, privacy protections, authentication, and verified provider data.