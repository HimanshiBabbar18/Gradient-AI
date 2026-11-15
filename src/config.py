"""Central configuration: paths, schema, target definition, UI labels."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SPD_DATA_DIR", ROOT / "data"))
ARTIFACT_DIR = Path(os.environ.get("SPD_ARTIFACT_DIR", ROOT / "artifacts"))

SEED = 42
TEST_SIZE = 0.20          # held-out test set (never touched during tuning)
VAL_SIZE = 0.15           # part of train used for early stopping of the final DNN

# Target: G3 (0-20) -> 3 employability grade bands
#   < 10   : below the Portuguese pass mark  -> Needs Improvement
#   10-14  : pass                            -> Good
#   >= 15  : high achiever                   -> Excellent
TARGET = "G3"
CLASS_NAMES = ["Needs Improvement", "Good", "Excellent"]
EMPLOYABILITY = {
    "Needs Improvement": ("Grade C", "At risk - academic support recommended before placement."),
    "Good": ("Grade B", "Employable - solid performance, room to move up."),
    "Excellent": ("Grade A", "Highly employable - strong academic profile."),
}


def grade_to_class(g: int) -> int:
    if g < 10:
        return 0
    if g < 15:
        return 1
    return 2


# Schema of the UCI dataset (30 inputs + G1, G2, G3)
NUMERIC = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
           "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"]
BINARY = ["school", "sex", "address", "famsize", "Pstatus", "schoolsup", "famsup",
          "paid", "activities", "nursery", "higher", "internet", "romantic"]
NOMINAL = ["Mjob", "Fjob", "reason", "guardian"]
ALL_FEATURES = NUMERIC + BINARY + NOMINAL
PRIOR_GRADES = ["G1", "G2"]

# Two experimental scenarios (key design decision, see README):
#   with_grades    : G1/G2 allowed            -> "mid-term" predictor
#   without_grades : G1/G2 dropped            -> "early-warning" predictor
SCENARIOS = {
    "with_grades": [],
    "without_grades": PRIOR_GRADES,   # columns to DROP
}

# Human friendly labels / help text for the Streamlit app
LABELS = {
    "school": ("School", "GP = Gabriel Pereira, MS = Mousinho da Silveira"),
    "sex": ("Sex", "F = female, M = male"),
    "age": ("Age", "Student age (15-22)"),
    "address": ("Home address type", "U = urban, R = rural"),
    "famsize": ("Family size", "LE3 = 3 or fewer members, GT3 = more than 3"),
    "Pstatus": ("Parents' cohabitation", "T = living together, A = apart"),
    "Medu": ("Mother's education", "0 none, 1 primary, 2 5th-9th grade, 3 secondary, 4 higher"),
    "Fedu": ("Father's education", "0 none, 1 primary, 2 5th-9th grade, 3 secondary, 4 higher"),
    "Mjob": ("Mother's job", "teacher / health / services / at_home / other"),
    "Fjob": ("Father's job", "teacher / health / services / at_home / other"),
    "reason": ("Reason for choosing school", "home / reputation / course / other"),
    "guardian": ("Guardian", "mother / father / other"),
    "traveltime": ("Home-to-school travel time", "1 <15 min, 2 15-30 min, 3 30-60 min, 4 >1 hour"),
    "studytime": ("Weekly study time", "1 <2 h, 2 2-5 h, 3 5-10 h, 4 >10 h"),
    "failures": ("Past class failures", "Number of past failures (3 means 3 or more)"),
    "schoolsup": ("Extra school support", "Extra educational support from school"),
    "famsup": ("Family educational support", "Support from family"),
    "paid": ("Paid extra classes", "Extra paid classes in the course subject"),
    "activities": ("Extra-curricular activities", "Takes part in extra-curricular activities"),
    "nursery": ("Attended nursery school", ""),
    "higher": ("Wants higher education", "Wants to take higher education"),
    "internet": ("Internet at home", ""),
    "romantic": ("In a relationship", ""),
    "famrel": ("Family relationship quality", "1 very bad ... 5 excellent"),
    "freetime": ("Free time after school", "1 very low ... 5 very high"),
    "goout": ("Going out with friends", "1 very low ... 5 very high"),
    "Dalc": ("Workday alcohol use", "1 very low ... 5 very high"),
    "Walc": ("Weekend alcohol use", "1 very low ... 5 very high"),
    "health": ("Current health status", "1 very bad ... 5 very good"),
    "absences": ("Number of school absences", "0-93"),
    "G1": ("First period grade (G1)", "0-20"),
    "G2": ("Second period grade (G2)", "0-20"),
}
