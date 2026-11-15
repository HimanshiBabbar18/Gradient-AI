# Gradient AI — Student Academic Performance Intelligence

**Predict Today. Improve Tomorrow.**

Gradient AI is a Streamlit application that uses saved Deep Neural Network (DNN) models to estimate student academic performance categories from historical student information. The app includes individual and bulk prediction, dataset exploration, model evaluation, and session-based prediction history.

## Features

- **Individual prediction:** enter one student's details and view the predicted performance category, confidence, class probabilities, and suggested support steps.
- **Two prediction scenarios:** mid-term prediction (G1 + G2 available) and early-warning prediction (G1 + G2 excluded).
- **Bulk prediction:** upload a CSV, validate required columns, predict multiple records, and download the output.
- **Dashboard:** review dataset size, average final grade, pass rate, and performance distributions.
- **Data insights:** explore selected numeric variables and saved EDA charts.
- **Model evaluation:** inspect DNN metrics, baseline comparisons, confusion matrices, training curves, ROC curves, and feature importance.
- **Prediction history:** review and export predictions made during the current session.

## Run locally on Windows

1. Extract the ZIP file.
2. Open Command Prompt in the `student_performance_app` folder.
3. Create a virtual environment:

```bat
python -m venv .venv
```

4. Activate it:

```bat
.venv\Scripts\activate
```

5. Install dependencies:

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Start the application:

```bat
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Project structure

```text
student_performance_app/
├── app.py                 # Streamlit user interface
├── train.py               # Model training and evaluation pipeline
├── requirements.txt       # Python dependencies
├── README.md              # Setup and project guide
├── data/                  # UCI Portuguese student dataset
├── src/                   # Data, DNN, baseline, selection and evaluation modules
└── artifacts/             # Saved models, preprocessors, metrics and plots
```

## Model and data notes

- The application loads the existing trained models and preprocessors from `artifacts/`. **Launching the app does not retrain models.**
- The included dataset is `data/student-por.csv` (Portuguese language subject). A Mathematics dataset is not included.
- The target is final grade `G3`, grouped as: **Needs Improvement** (below 10), **Good** (10–14), and **Excellent** (15 or above).
- The mid-term model can use `G1` and `G2`; the early-warning model excludes both.
- Bulk CSV uploads must contain the raw feature columns listed on the Bulk Prediction page. Do not one-hot encode them manually.
- Predictions are estimates from historical data, not guarantees or diagnoses. They should support—not replace—teacher judgement and student context.

## Retraining (optional)

Only retrain when you intentionally want to regenerate the saved artifacts. From the project root:

```bat
python train.py --quick
```

For the full training and evaluation pipeline:

```bat
python train.py
```

Training may overwrite files in `artifacts/`. Keep a backup if you need to preserve the current model results.
