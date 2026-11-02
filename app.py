from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
app.secret_key = "carepoint-demo-key-change-before-deployment"
DB_PATH = Path(__file__).with_name("hospital.db")

# Demo dataset for an educational NLP recommendation prototype.
# It recommends a department/specialist; it does NOT diagnose disease.
TRAINING_EXAMPLES = [
    ("tooth pain toothache cavity gum swelling bleeding teeth dental pain", "Dentist"),
    ("broken tooth sensitive teeth wisdom tooth jaw dental checkup", "Dentist"),
    ("skin rash acne pimples itching eczema psoriasis skin spots", "Dermatologist"),
    ("hair loss scalp itching skin allergy unusual mole", "Dermatologist"),
    ("bone fracture broken bone joint pain knee pain shoulder pain back pain", "Orthopedic Doctor"),
    ("sprain ankle injury joint stiffness bone pain", "Orthopedic Doctor"),
    ("heart chest discomfort palpitations high blood pressure heart disease", "Cardiologist"),
    ("shortness of breath heart rhythm chest pain", "Cardiologist"),
    ("eye pain blurry vision red eyes vision problem", "Ophthalmologist"),
    ("difficulty seeing eye infection cataract", "Ophthalmologist"),
    ("ear pain hearing loss sore throat blocked nose sinus", "ENT Specialist"),
    ("tonsils voice problem nose bleeding ear nose throat", "ENT Specialist"),
    ("stomach pain acidity indigestion bloating diarrhea constipation", "Gastroenterologist"),
    ("digestive issue abdominal discomfort liver intestine problem", "Gastroenterologist"),
    ("kidney pain kidney stones urine problem urinary changes", "Nephrologist"),
    ("kidney disease abnormal kidney function swelling", "Nephrologist"),
    ("headache migraine seizure tremor numbness nerve problem", "Neurologist"),
    ("memory problem dizziness nervous system weakness", "Neurologist"),
    ("anxiety depression panic mood problem emotional distress", "Psychiatrist"),
    ("sleep problems severe stress mental health support", "Psychiatrist"),
    ("pregnancy menstrual periods reproductive health women's health", "Obstetrician/Gynecologist"),
    ("pregnancy checkup irregular periods pelvic pain", "Obstetrician/Gynecologist"),
    ("child fever child cough infant baby pediatric checkup", "Pediatrician"),
    ("child growth vaccination newborn health", "Pediatrician"),
    ("fever cold cough flu general weakness common illness", "General Physician"),
    ("general checkup body ache mild fever fatigue", "General Physician"),
    ("cancer oncology tumor cancer treatment chemotherapy", "Oncologist"),
    ("known cancer diagnosis oncology follow up tumor specialist", "Oncologist"),
    ("blood test anemia unusual bruising blood disorder", "Hematologist"),
    ("diabetes high blood sugar thyroid hormone endocrine", "Endocrinologist"),
]

DOCTORS = {
    "Dentist": [
        {"name": "Dr. Aditi Sharma", "qualification": "BDS", "experience": "8 years"},
        {"name": "Dr. Rohan Mehta", "qualification": "MDS", "experience": "6 years"},
    ],
    "Dermatologist": [
        {"name": "Dr. Neha Kulkarni", "qualification": "MD Dermatology", "experience": "9 years"},
        {"name": "Dr. Ishan Patel", "qualification": "DVD", "experience": "5 years"},
    ],
    "Orthopedic Doctor": [
        {"name": "Dr. Arjun Deshmukh", "qualification": "MS Orthopedics", "experience": "11 years"},
        {"name": "Dr. Priya Nair", "qualification": "DNB Orthopedics", "experience": "7 years"},
    ],
    "Cardiologist": [
        {"name": "Dr. Karan Shah", "qualification": "DM Cardiology", "experience": "12 years"},
        {"name": "Dr. Meera Joshi", "qualification": "MD, DM Cardiology", "experience": "8 years"},
    ],
    "Ophthalmologist": [
        {"name": "Dr. Kavya Rao", "qualification": "MS Ophthalmology", "experience": "10 years"},
        {"name": "Dr. Nikhil Bhat", "qualification": "DOMS", "experience": "6 years"},
    ],
    "ENT Specialist": [
        {"name": "Dr. Sameer Khan", "qualification": "MS ENT", "experience": "9 years"},
        {"name": "Dr. Pooja Menon", "qualification": "DLO", "experience": "5 years"},
    ],
    "Gastroenterologist": [
        {"name": "Dr. Vivek Sane", "qualification": "DM Gastroenterology", "experience": "10 years"},
        {"name": "Dr. Ananya Rao", "qualification": "MD, DM Gastroenterology", "experience": "7 years"},
    ],
    "Nephrologist": [
        {"name": "Dr. Siddharth Patil", "qualification": "DM Nephrology", "experience": "8 years"},
        {"name": "Dr. Ritu Verma", "qualification": "MD, DM Nephrology", "experience": "6 years"},
    ],
    "Neurologist": [
        {"name": "Dr. Rahul Desai", "qualification": "DM Neurology", "experience": "11 years"},
        {"name": "Dr. Sana Qureshi", "qualification": "MD, DM Neurology", "experience": "7 years"},
    ],
    "Psychiatrist": [
        {"name": "Dr. Nandita Shah", "qualification": "MD Psychiatry", "experience": "9 years"},
        {"name": "Dr. Adil Fernandes", "qualification": "DPM", "experience": "6 years"},
    ],
    "Obstetrician/Gynecologist": [
        {"name": "Dr. Shreya Deshpande", "qualification": "MS Obstetrics & Gynecology", "experience": "10 years"},
        {"name": "Dr. Farah Ali", "qualification": "DNB Obstetrics & Gynecology", "experience": "7 years"},
    ],
    "Pediatrician": [
        {"name": "Dr. Manasi Gokhale", "qualification": "MD Pediatrics", "experience": "10 years"},
        {"name": "Dr. Omkar Kulkarni", "qualification": "DCH", "experience": "6 years"},
    ],
    "General Physician": [
        {"name": "Dr. Akash More", "qualification": "MBBS, MD Medicine", "experience": "12 years"},
        {"name": "Dr. Sneha Iyer", "qualification": "MBBS", "experience": "7 years"},
    ],
    "Oncologist": [
        {"name": "Dr. Anil Kapoor", "qualification": "DM Medical Oncology", "experience": "13 years"},
        {"name": "Dr. Rhea Sinha", "qualification": "MD, DM Oncology", "experience": "8 years"},
    ],
    "Hematologist": [
        {"name": "Dr. Devika Rao", "qualification": "DM Hematology", "experience": "9 years"},
    ],
    "Endocrinologist": [
        {"name": "Dr. Mihir Joshi", "qualification": "DM Endocrinology", "experience": "10 years"},
    ],
}

def get_ai_recommender():
    texts = [row[0] for row in TRAINING_EXAMPLES]
    labels = [row[1] for row in TRAINING_EXAMPLES]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, labels

VECTORIZER, TRAINING_MATRIX, TRAINING_LABELS = get_ai_recommender()

def recommend_specialist(problem):
    cleaned = (problem or "").strip()
    if not cleaned:
        return None, 0.0
    query_vec = VECTORIZER.transform([cleaned])
    scores = cosine_similarity(query_vec, TRAINING_MATRIX).flatten()
    best_idx = int(scores.argmax())
    score = float(scores[best_idx])
    if score < 0.08:
        return "General Physician", score
    return TRAINING_LABELS[best_idx], score

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                email TEXT NOT NULL,
                problem TEXT NOT NULL,
                specialist TEXT NOT NULL,
                doctor_name TEXT NOT NULL,
                appointment_date TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

@app.route("/", methods=["GET", "POST"])
def index():
    recommendation = None
    doctors = []
    problem = ""
    score_pct = None
    if request.method == "POST":
        problem = request.form.get("problem", "").strip()
        if len(problem) < 3:
            flash("Please describe the health concern in at least a few words.", "error")
        else:
            specialist, score = recommend_specialist(problem)
            recommendation = specialist
            doctors = DOCTORS.get(specialist, DOCTORS["General Physician"])
            score_pct = round(score * 100)
            if score < 0.08:
                flash("The AI could not find a close match, so it suggests starting with a general physician.", "info")
    return render_template("index.html", recommendation=recommendation, doctors=doctors,
                           problem=problem, score_pct=score_pct)

@app.route("/book", methods=["POST"])
def book():
    patient_name = request.form.get("patient_name", "").strip()
    email = request.form.get("email", "").strip()
    problem = request.form.get("problem", "").strip()
    specialist = request.form.get("specialist", "").strip()
    doctor_name = request.form.get("doctor_name", "").strip()
    appointment_date = request.form.get("appointment_date", "").strip()

    if not all([patient_name, email, problem, specialist, doctor_name, appointment_date]):
        flash("Please fill in all appointment fields.", "error")
        return redirect(url_for("index"))

    # The appointment is a demo booking; no real clinic or doctor is connected.
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("""
            INSERT INTO appointments
            (patient_name, email, problem, specialist, doctor_name, appointment_date, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (patient_name, email, problem, specialist, doctor_name, appointment_date,
              datetime.now().isoformat(timespec="minutes")))
        appointment_id = cur.lastrowid
        conn.commit()
    return redirect(url_for("confirmation", appointment_id=appointment_id))

@app.route("/confirmation/<int:appointment_id>")
def confirmation(appointment_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        appointment = conn.execute(
            "SELECT * FROM appointments WHERE id = ?", (appointment_id,)
        ).fetchone()
    if appointment is None:
        flash("Appointment not found.", "error")
        return redirect(url_for("index"))
    return render_template("confirmation.html", appointment=appointment)

@app.route("/appointments")
def appointments():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM appointments ORDER BY id DESC").fetchall()
    return render_template("appointments.html", appointments=rows)

@app.route("/doctors")
def doctors_page():
    return render_template("doctors.html", doctors=DOCTORS)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)