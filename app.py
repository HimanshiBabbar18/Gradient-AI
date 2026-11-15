"""
Gradient AI — Student Academic Performance Intelligence
Run from the project root: streamlit run app.py

Uses the already-trained models in artifacts/. It does NOT retrain them.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config as C

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="Gradient AI | Student Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------- Gradient AI visual theme --------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root { --ink:#eaf0ff; --muted:#9eacc7; --line:rgba(148,163,184,.17); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp {
  background:
    radial-gradient(ellipse at 12% 0%, rgba(48,91,170,.20), transparent 34%),
    radial-gradient(ellipse at 95% 8%, rgba(20,160,140,.10), transparent 28%),
    #080f1d;
  color:var(--ink);
}
[data-testid="stHeader"] { background:rgba(8,15,29,.72); }
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#101c31 0%,#0b1425 100%);
  border-right:1px solid var(--line);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  font-family:'Manrope',sans-serif; letter-spacing:-.5px;
}
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.7px; }
h1 { font-weight:800 !important; }
p, label, .stCaption { color:#c0cbe0; }
[data-testid="stMetric"] {
  background:linear-gradient(145deg,rgba(26,43,70,.92),rgba(15,27,46,.92));
  border:1px solid rgba(132,163,214,.20);
  padding:18px 20px; border-radius:18px;
  box-shadow:0 10px 28px rgba(0,0,0,.12);
}
[data-testid="stMetricLabel"] { color:#a9bad7 !important; }
[data-testid="stMetricValue"] { color:#f4f7ff !important; font-family:'Manrope',sans-serif; }
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-color:var(--line) !important; border-radius:18px;
}
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
  border:0; border-radius:12px; min-height:44px; font-weight:700;
  background:linear-gradient(100deg,#3978ed,#5b9bff);
  color:white; transition:all .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform:translateY(-1px); box-shadow:0 8px 22px rgba(57,120,237,.25);
  color:white; border:0;
}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
  background:#14233a; border-color:#344967; border-radius:10px;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border:1px solid var(--line); border-radius:14px; overflow:hidden;
}
hr { border-color:var(--line); }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] {
  background:#14233a; border-radius:10px 10px 0 0; padding:10px 16px;
}
.stAlert { border-radius:14px; }
div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:14px; }
small { color:var(--muted); }
.brand-tagline { color:#8fb5ff; font-size:10px; font-weight:800; letter-spacing:1.25px; margin: -7px 0 18px 2px; }
.hero-panel { padding:26px 28px; border:1px solid rgba(100,151,245,.25); border-radius:20px;
  background:linear-gradient(115deg,rgba(40,86,160,.30),rgba(20,36,62,.78));
  box-shadow:0 16px 42px rgba(0,0,0,.16); margin:4px 0 24px; }
.hero-kicker { color:#91b9ff; font-size:11px; font-weight:800; letter-spacing:1.6px; text-transform:uppercase; }
.hero-title { color:#f4f7ff; font-family:'Manrope',sans-serif; font-size:30px; font-weight:800; line-height:1.15; margin:8px 0; }
.hero-copy { color:#b8c8e4; font-size:14px; margin:0; }
.section-note { color:#91a6c7; font-size:13px; margin-top:-8px; margin-bottom:16px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  padding:10px 12px; margin:3px 0; border:1px solid transparent;
  border-radius:12px; transition:all .18s ease; cursor:pointer;
  background:rgba(255,255,255,.015);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background:rgba(75,132,225,.13); border-color:rgba(115,164,245,.20);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background:linear-gradient(100deg,rgba(57,120,237,.24),rgba(91,155,255,.10));
  border-color:rgba(112,169,255,.38); box-shadow:inset 3px 0 0 #6da6ff;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
  color:#dce8ff; font-weight:600; font-size:14px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
  color:#ffffff; font-weight:700;
}
[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,#3978ed,#70a9ff); }
.stDownloadButton > button { width:100%; }
@media (max-width: 700px) { .hero-panel { padding:20px; } .hero-title { font-size:24px; } }

</style>
""", unsafe_allow_html=True)

SCENARIO_TITLES = {
    "with_grades": "Mid-term prediction (G1 + G2 available)",
    "without_grades": "Early-warning prediction (no previous grades)",
}
CLASS_COLORS = {
    "Needs Improvement": "#EF4444",
    "Good": "#F59E0B",
    "Excellent": "#10B981",
}


def scenario_available(scenario: str) -> bool:
    folder = ARTIFACTS / scenario
    return all((folder / f).exists() for f in ["dnn_model.keras", "preprocessor.joblib", "meta.json"])


@st.cache_resource(show_spinner="Loading trained model...")
def load_artifacts(scenario: str):
    import tensorflow as tf
    folder = ARTIFACTS / scenario
    model = tf.keras.models.load_model(folder / "dnn_model.keras")
    preprocessor = joblib.load(folder / "preprocessor.joblib")
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    return model, preprocessor, meta


@st.cache_data
def load_dataset():
    path = DATA_DIR / "student-por.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=";")
    if "grade_class" not in df.columns and "G3" in df.columns:
        df["grade_class"] = df["G3"].apply(C.grade_to_class)
    return df


def predict_rows(rows: pd.DataFrame, model, prep, meta):
    """Predict from raw feature columns in the same order used during training."""
    selected = meta["selected_features"]
    missing = [col for col in selected if col not in rows.columns]
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(missing))
    clean_rows = rows[selected].copy()
    probabilities = model.predict(prep.transform(clean_rows), verbose=0)
    indices = probabilities.argmax(axis=1)
    output = rows.copy()
    output["Predicted category"] = [C.CLASS_NAMES[int(i)] for i in indices]
    output["Confidence (%)"] = (probabilities.max(axis=1) * 100).round(2)
    for i, name in enumerate(C.CLASS_NAMES):
        output[f"P({name}) (%)"] = (probabilities[:, i] * 100).round(2)
    return output


def recommendation(label: str, values: dict) -> list[str]:
    tips = []
    if label == "Needs Improvement":
        tips += [
            "Create a weekly study timetable and review progress every week.",
            "Arrange subject-specific support or tutoring for difficult topics.",
            "Set small, measurable academic goals and monitor attendance."
        ]
    elif label == "Good":
        tips += [
            "Maintain the current study routine and revise consistently.",
            "Focus extra practice on the subjects/topics with the lowest scores.",
            "Use periodic mock tests to identify gaps before final assessments."
        ]
    else:
        tips += [
            "Maintain consistent study habits and avoid last-minute preparation.",
            "Use advanced exercises or projects to deepen subject understanding.",
            "Support peers through group study while protecting personal study time."
        ]
    if values.get("absences", 0) >= 10:
        tips.append("Attendance may need attention: identify the reasons for frequent absences.")
    if values.get("failures", 0) >= 1:
        tips.append("Review previously failed subjects first and seek help early.")
    if values.get("studytime", 0) <= 1:
        tips.append("Try gradually increasing focused weekly study time.")
    # De-duplicate while preserving order.
    return list(dict.fromkeys(tips))


def metric_value(metrics: dict, key: str):
    value = metrics.get(key)
    return "—" if value is None else f"{value * 100:.1f}%"


def render_image(folder: Path, filename: str, caption: str, width="stretch"):
    path = folder / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=(width == "stretch"))


def show_dashboard(df: pd.DataFrame | None, scenario: str, meta: dict):
    st.markdown("""<div class="hero-panel"><div class="hero-kicker">Student intelligence platform</div>
    <div class="hero-title">Predict today. Improve tomorrow.</div>
    <p class="hero-copy">Explore academic patterns, estimate performance bands, and turn model insights into practical student support.</p></div>""", unsafe_allow_html=True)
    if df is None:
        st.warning("Dataset file not found at data/student-por.csv. Prediction still works, but dataset analytics are unavailable.")
        return

    total = len(df)
    avg = df["G3"].mean() if "G3" in df else None
    pass_rate = (df["G3"].ge(10).mean() * 100) if "G3" in df else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students in dataset", f"{total:,}")
    c2.metric("Average final grade (G3)", f"{avg:.2f}/20" if avg is not None else "—")
    c3.metric("Pass rate", f"{pass_rate:.1f}%" if pass_rate is not None else "—")
    c4.metric("Selected model features", len(meta["selected_features"]))

    left, right = st.columns(2)
    with left:
        st.subheader("Performance category distribution")
        if "grade_class" in df:
            counts = df["grade_class"].map(dict(enumerate(C.CLASS_NAMES))).value_counts()
            st.bar_chart(counts)
    with right:
        st.subheader("Final grade distribution")
        if "G3" in df:
            hist = df["G3"].value_counts().sort_index()
            st.bar_chart(hist)

    st.subheader("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    st.caption("Dataset: UCI Student Performance — Portuguese language subject. This is a historical dataset, not live student records.")


def show_single_prediction(scenario: str, model, prep, meta):
    st.title("🔮 Individual Student Prediction")
    st.markdown(f"<p class='section-note'>{SCENARIO_TITLES[scenario]} · Enter student details to generate an estimate.</p>", unsafe_allow_html=True)
    st.info("Enter the details below. Only features selected by the trained model are requested.")
    values = {}
    features = meta["features"]
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            name = feature["name"]
            if feature["type"] == "numeric":
                values[name] = st.number_input(
                    feature["label"], min_value=int(feature["min"]), max_value=int(feature["max"]),
                    value=int(feature["default"]), step=1, help=feature.get("help") or None,
                    key=f"{scenario}_{name}",
                )
            else:
                options = feature["options"]
                default = feature.get("default", options[0])
                values[name] = st.selectbox(
                    feature["label"], options, index=options.index(default),
                    help=feature.get("help") or None, key=f"{scenario}_{name}",
                )

    if st.button("Predict academic performance", type="primary", use_container_width=True):
        row = pd.DataFrame([values])
        result = predict_rows(row, model, prep, meta)
        label = result.iloc[0]["Predicted category"]
        confidence = float(result.iloc[0]["Confidence (%)"])
        st.session_state["last_prediction"] = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario,
            "values": values,
            "category": label,
            "confidence": confidence,
            "probabilities": {name: float(result.iloc[0][f"P({name}) (%)"]) for name in C.CLASS_NAMES},
        }
        st.session_state.setdefault("prediction_history", []).insert(0, {
            "Timestamp": st.session_state["last_prediction"]["time"],
            "Scenario": scenario,
            "Predicted category": label,
            "Confidence (%)": confidence,
            **values,
        })

    last = st.session_state.get("last_prediction")
    if last and last.get("scenario") == scenario:
        st.divider()
        label = last["category"]
        grade, note = C.EMPLOYABILITY[label]
        st.subheader("Prediction result")
        a, b, c = st.columns(3)
        a.metric("Predicted category", label)
        b.metric("Confidence", f"{last['confidence']:.1f}%")
        c.metric("Performance band", grade)
        if label == "Needs Improvement":
            st.error(note)
        elif label == "Good":
            st.warning(note)
        else:
            st.success(note)

        st.subheader("Class probability")
        probs = pd.DataFrame({"Probability (%)": last["probabilities"]})
        st.bar_chart(probs)
        st.subheader("Suggested next steps")
        for tip in recommendation(label, last["values"]):
            st.write("• " + tip)
        st.caption("Predictions are estimates from the trained model, not a diagnosis or a guarantee of future results.")


def show_bulk_prediction(scenario: str, model, prep, meta):
    st.title("📂 Bulk Student Prediction")
    st.markdown("<p class='section-note'>Run predictions for multiple students in one go and export the results.</p>", unsafe_allow_html=True)
    st.caption("The CSV should contain raw values (for example GP/MS, yes/no, and numeric values), not one-hot encoded columns.")
    required = meta["selected_features"]
    st.code(", ".join(required))
    uploaded = st.file_uploader("Upload student CSV", type=["csv"], key=f"bulk_{scenario}")
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            st.write("Uploaded data preview")
            st.dataframe(data.head(), use_container_width=True)
            missing = [col for col in required if col not in data.columns]
            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            elif st.button("Predict all students", type="primary"):
                result = predict_rows(data, model, prep, meta)
                st.success(f"Predictions completed for {len(result)} students.")
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "Download predictions CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name=f"student_predictions_{scenario}.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Could not read/process this CSV: {exc}")


def show_insights(df: pd.DataFrame | None):
    st.title("📈 Data Insights & EDA")
    st.markdown("<p class='section-note'>Explore distributions, relationships, and patterns in the supplied student dataset.</p>", unsafe_allow_html=True)
    if df is None:
        st.warning("Dataset not found. Add data/student-por.csv to enable this page.")
        return
    numeric_cols = [c for c in ["G1", "G2", "G3", "absences", "studytime", "failures", "age"] if c in df.columns]
    selected = st.multiselect("Choose numeric columns", numeric_cols, default=[c for c in ["G1", "G2", "G3"] if c in numeric_cols])
    if selected:
        st.subheader("Interactive distributions")
        for col in selected:
            st.write(f"**{col}**")
            st.bar_chart(df[col].value_counts().sort_index())

    st.subheader("Saved exploratory data analysis")
    image_files = [
        ("01_g3_distribution.png", "Final grade (G3) distribution"),
        ("02_class_balance.png", "Performance class balance"),
        ("03_correlation.png", "Feature correlation"),
        ("04_g3_vs_key_features.png", "G3 vs key features"),
        ("05_absences_g2_vs_g3.png", "Absences, G2 and G3"),
        ("06_class_share_by_category.png", "Class share by category"),
    ]
    cols = st.columns(2)
    for i, (filename, caption) in enumerate(image_files):
        path = ARTIFACTS / "eda" / filename
        if path.exists():
            with cols[i % 2]:
                st.image(str(path), caption=caption, use_container_width=True)


def show_model_performance(scenario: str, meta: dict):
    st.title("🏆 Model Performance")
    st.markdown("<p class='section-note'>Review held-out evaluation metrics and compare the trained DNN with baseline models.</p>", unsafe_allow_html=True)
    folder = ARTIFACTS / scenario
    st.write(SCENARIO_TITLES[scenario])
    metrics = meta.get("test_metrics", {})
    dnn_metrics = metrics.get("DNN (Keras)", {})
    a, b, c, d = st.columns(4)
    a.metric("DNN accuracy", metric_value(dnn_metrics, "accuracy"))
    b.metric("Macro F1", metric_value(dnn_metrics, "f1_macro"))
    c.metric("Weighted F1", metric_value(dnn_metrics, "f1_weighted"))
    d.metric("ROC-AUC (OvR)", metric_value(dnn_metrics, "roc_auc_ovr"))

    comparison_path = folder / "comparison.csv"
    if comparison_path.exists():
        st.subheader("Traditional ML vs DNN")
        comp = pd.read_csv(comparison_path, index_col=0)
        st.dataframe(comp.style.format("{:.3f}"), use_container_width=True)
    else:
        st.warning("comparison.csv is not present for this scenario.")

    for filename, caption in [
        ("comparison.png", "Model comparison"),
        ("confusion_dnn.png", "DNN confusion matrix"),
        ("confusion_rf.png", "Random Forest confusion matrix"),
        ("training_curves.png", "DNN training curves"),
        ("roc_dnn.png", "DNN ROC curves"),
        ("permutation_importance.png", "Permutation feature importance"),
    ]:
        render_image(folder, filename, caption)

    st.subheader("Model configuration")
    st.write("**Selected features:** " + ", ".join(meta.get("selected_features", [])))
    st.json(meta.get("best_hyperparameters", {}))
    st.caption(f"Held-out test set: {meta.get('n_test', '—')} records; training set: {meta.get('n_train', '—')} records.")


def show_history():
    st.title("🕘 Prediction History")
    st.markdown("<p class='section-note'>Predictions created during this app session appear here.</p>", unsafe_allow_html=True)
    history = st.session_state.get("prediction_history", [])
    if not history:
        st.info("No predictions have been made in this session yet.")
        return
    frame = pd.DataFrame(history)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        "Download prediction history",
        frame.to_csv(index=False).encode("utf-8"),
        file_name="prediction_history.csv",
        mime="text/csv",
    )
    if st.button("Clear session history"):
        st.session_state["prediction_history"] = []
        st.session_state.pop("last_prediction", None)
        st.rerun()


def show_about():
    st.title("ℹ️ About Gradient AI")
    st.markdown("""
    **Gradient AI — Student Academic Performance Prediction** uses a Deep Neural Network (DNN)
    to estimate a student's final-grade category from historical student information.

    - **Needs Improvement:** final grade below 10
    - **Good:** final grade from 10 to 14
    - **Excellent:** final grade 15 or above
    - **Mid-term model:** includes G1 and G2.
    - **Early-warning model:** excludes G1 and G2.

    The project also compares the DNN with Logistic Regression, Random Forest,
    and Gradient Boosting. It uses the saved model artifacts; opening this app
    does not start model training.

    **Important:** the dataset is the UCI Student Performance dataset for the
    Portuguese language subject. Predictions are educational estimates and
    should not be used as the sole basis for decisions about a student.

    **Tagline:** *Predict Today. Improve Tomorrow.*
    """)


# Sidebar navigation and model selection.
available_scenarios = [s for s in C.SCENARIOS if scenario_available(s)]
if not available_scenarios:
    st.error("Trained model files are missing. Expected artifacts/<scenario>/dnn_model.keras, preprocessor.joblib and meta.json.")
    st.stop()

st.sidebar.markdown("## 🧠 Gradient AI")
st.sidebar.markdown("<div class='brand-tagline'>PREDICT TODAY. IMPROVE TOMORROW.</div>", unsafe_allow_html=True)
NAV_ITEMS = {
    "⌂  Dashboard": "Dashboard",
    "◉  Single Prediction": "Single Prediction",
    "▤  Bulk Prediction": "Bulk Prediction",
    "▥  Data Insights": "Data Insights",
    "✧  Model Performance": "Model Performance",
    "◷  Prediction History": "Prediction History",
    "ⓘ  About Gradient AI": "About",
}
selected_nav = st.sidebar.radio(
    "NAVIGATION",
    list(NAV_ITEMS.keys()),
    label_visibility="visible",
)
page = NAV_ITEMS[selected_nav]
scenario = st.sidebar.selectbox(
    "Prediction model",
    available_scenarios,
    format_func=lambda x: SCENARIO_TITLES[x],
)
st.sidebar.divider()
st.sidebar.caption("Models are loaded from saved artifacts. No retraining is performed.")

model, prep, meta = load_artifacts(scenario)
dataset = load_dataset()

if page == "Dashboard":
    show_dashboard(dataset, scenario, meta)
elif page == "Single Prediction":
    show_single_prediction(scenario, model, prep, meta)
elif page == "Bulk Prediction":
    show_bulk_prediction(scenario, model, prep, meta)
elif page == "Data Insights":
    show_insights(dataset)
elif page == "Model Performance":
    show_model_performance(scenario, meta)
elif page == "Prediction History":
    show_history()
else:
    show_about()
