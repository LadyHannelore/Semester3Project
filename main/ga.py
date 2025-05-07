import gradio as gr
import pandas as pd
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import base64
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import os
from ortools.sat.python import cp_model
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.spx import SinglePointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Configuration & Setup ---
warnings.filterwarnings("ignore", category=UserWarning)
try:
    plt.style.use("seaborn-white")
except OSError:
    plt.style.use("ggplot")

NUM_STUDENTS = 1000
CLASS_SIZE_TARGET = 30
N_CLUSTERS = max(1, NUM_STUDENTS // CLASS_SIZE_TARGET)
SYNTHETIC_DATA_CSV = "synthetic_student_data.csv"

# --- Updated Constraints ---
MAX_ALLOWED_DIFFERENCE = 15  # Relaxed from 10 to 15
MAX_ALLOWED_WELLBEING_DIFF = 2  # Relaxed from 1 to 2
MAX_BULLIES_PER_CLASS = 2  # Relaxed from 1 to 2

# --- Data Generation ---
def generate_synthetic_data(filename=SYNTHETIC_DATA_CSV, num_students=NUM_STUDENTS):
    """Generates synthetic student data if the CSV doesn't exist."""
    if os.path.exists(filename):
        print(f"Loading existing data from {filename}")
        try:
            df = pd.read_csv(filename)
            if len(df) == num_students:
                required_cols = [
                    'StudentID', 'Academic_Performance', 'Friends', 'bullying',
                    'Student_ID', 'Wellbeing_Score', 'Bullying_Score'
                ]
                if all(col in df.columns for col in required_cols):
                    print("Data loaded successfully.")
                    return df
                else:
                    print("Required columns missing in CSV, using as-is.")
                    # Add missing columns with default values if not present
                    if 'Student_ID' not in df.columns:
                        df['Student_ID'] = df['StudentID'] if 'StudentID' in df.columns else [f"S{i:04d}" for i in range(1, len(df)+1)]
                    if 'Wellbeing_Score' not in df.columns:
                        if 'k6_overall' in df.columns:
                            df['Wellbeing_Score'] = df['k6_overall']
                        else:
                            df['Wellbeing_Score'] = 0
                    if 'Bullying_Score' not in df.columns:
                        if 'bullying' in df.columns:
                            df['Bullying_Score'] = df['bullying']
                        else:
                            df['Bullying_Score'] = 0
                    return df
            else:
                print(f"CSV found but has {len(df)} rows, expected {num_students}. Using as-is.")
                # Add missing columns with default values if not present
                if 'Student_ID' not in df.columns:
                    df['Student_ID'] = df['StudentID'] if 'StudentID' in df.columns else [f"S{i:04d}" for i in range(1, len(df)+1)]
                if 'Wellbeing_Score' not in df.columns:
                    if 'k6_overall' in df.columns:
                        df['Wellbeing_Score'] = df['k6_overall']
                    else:
                        df['Wellbeing_Score'] = 0
                if 'Bullying_Score' not in df.columns:
                    if 'bullying' in df.columns:
                        df['Bullying_Score'] = df['bullying']
                    else:
                        df['Bullying_Score'] = 0
                return df
        except Exception as e:
            print(f"Error loading CSV: {e}. Using as-is if possible.")
            try:
                df = pd.read_csv(filename)
                # Add missing columns with default values if not present
                if 'Student_ID' not in df.columns:
                    df['Student_ID'] = df['StudentID'] if 'StudentID' in df.columns else [f"S{i:04d}" for i in range(1, len(df)+1)]
                if 'Wellbeing_Score' not in df.columns:
                    if 'k6_overall' in df.columns:
                        df['Wellbeing_Score'] = df['k6_overall']
                    else:
                        df['Wellbeing_Score'] = 0
                if 'Bullying_Score' not in df.columns:
                    if 'bullying' in df.columns:
                        df['Bullying_Score'] = df['bullying']
                    else:
                        df['Bullying_Score'] = 0
                return df
            except Exception as e2:
                print(f"Failed to load CSV at all: {e2}. Regenerating...")

    print(f"Generating {num_students} new synthetic student records...")
    LIKERT_SCALE_1_7 = list(range(1, 8))
    K6_SCALE_1_5 = list(range(1, 6))
    LANGUAGE_SCALE = [0, 1]
    PWI_SCALE = list(range(0, 11))

    student_ids = [f"S{i:04d}" for i in range(1, num_students + 1)]
    data = []

    # --- Assign exactly 2 bullies per 40 students ---
    bully_indices = set()
    group_size = 40
    num_groups = num_students // group_size + (1 if num_students % group_size else 0)
    for group in range(num_groups):
        start = group * group_size
        end = min((group + 1) * group_size, num_students)
        group_indices = list(range(start, end))
        if len(group_indices) >= 2:
            selected = random.sample(group_indices, 2)
            bully_indices.update(selected)
        elif len(group_indices) == 1:
            bully_indices.update(group_indices)  # If only one left, assign as bully

    for idx, student_id in enumerate(student_ids):
        # --- Introduce more outliers ---
        # 10% chance for extreme academic performance
        if random.random() < 0.1:
            academic_performance = random.choice([random.randint(0, 40), random.randint(90, 100)])
        else:
            academic_performance = max(0, min(100, round(np.random.normal(70, 20))))  # wider stddev

        # 5% chance for extreme wellbeing (k6)
        if random.random() < 0.05:
            k6_scores = {f"k6_{i}": random.choice([1, 5]) for i in range(1, 7)}
        else:
            k6_scores = {f"k6_{i}": random.choice(K6_SCALE_1_5) for i in range(1, 7)}

        # --- Assign bullying: only 2 per 40 students ---
        if idx in bully_indices:
            bullying = 7  # Highest bullying score
        else:
            bullying = random.choice([1, 2, 3, 4, 5])  # Lower bullying scores

        # --- Generate more friendships ---
        # 10% chance to have many friends (10-20), 10% chance to have none, else 4-10 friends
        if random.random() < 0.1:
            num_friends = random.randint(10, 20)
        elif random.random() < 0.1:
            num_friends = 0
        else:
            num_friends = random.randint(4, 10)
        possible_friends = [pid for pid in student_ids if pid != student_id]
        friends = ", ".join(random.sample(possible_friends, k=min(num_friends, len(possible_friends))))

        manbox5_scores = {f"Manbox5_{i}": random.choice(LIKERT_SCALE_1_7) for i in range(1, 6)}

        student_data = {
            "StudentID": student_id,
            "Academic_Performance": academic_performance,
            "isolated": random.choice(LIKERT_SCALE_1_7),
            "WomenDifferent": random.choice(LIKERT_SCALE_1_7),
            "language": random.choices(LANGUAGE_SCALE, weights=[0.8, 0.2], k=1)[0],
            "COVID": random.choice(LIKERT_SCALE_1_7),
            "criticises": random.choice(LIKERT_SCALE_1_7),
            "MenBetterSTEM": random.choice(LIKERT_SCALE_1_7),
            "pwi_wellbeing": random.choice(PWI_SCALE),
            "Intelligence1": random.choice(LIKERT_SCALE_1_7),
            "Intelligence2": random.choice(LIKERT_SCALE_1_7),
            "Soft": random.choice(LIKERT_SCALE_1_7),
            "opinion": random.choice(LIKERT_SCALE_1_7),
            "Nerds": random.choice(LIKERT_SCALE_1_7),
            "comfortable": random.choice(LIKERT_SCALE_1_7),
            "future": random.choice(LIKERT_SCALE_1_7),
            "bullying": bullying,
            "Friends": friends,
            **manbox5_scores,
            **k6_scores,
            "Disrespect": random.randint(0, 10),  # Default range for Disrespect
            "K6_Score": random.randint(0, 24),  # Default range for K6 Score
            "Anxiety_Level": random.randint(0, 10),  # Default range for Anxiety Level
            "Depression_Level": random.randint(0, 10),  # Default range for Depression Level,
        }
        # Add required columns for upload page compatibility
        student_data["Student_ID"] = student_id
        student_data["Wellbeing_Score"] = sum(k6_scores.values())
        student_data["Bullying_Score"] = bullying
        data.append(student_data)

    df = pd.DataFrame(data)

    # Ensure all optional columns are present
    optional_columns = ["Disrespect", "K6_Score", "Anxiety_Level", "Depression_Level"]
    for col in optional_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing optional columns with default values

    df['School_support_engage'] = df[['criticises', 'comfortable', 'bullying', 'future']].mean(axis=1)

    df.to_csv(filename, index=False)
    print(f"Synthetic data saved to {filename}")
    return df

# --- Predictive Analytics ---
def run_analysis(df):
    """Performs prediction on the dataframe (no clustering)."""
    print("Running predictive analytics...")
    df['Academic_Success'] = (df['Academic_Performance'] > df['Academic_Performance'].quantile(0.75)).astype(int)
    df['Wellbeing_Decline'] = (df['Wellbeing_Score'] > df['Wellbeing_Score'].quantile(0.75)).astype(int)
    df['Friends_Count'] = df['Friends'].fillna('').apply(lambda x: len(x.split(', ')) if x else 0)
    df['Positive_Peer_Collab'] = (df['Friends_Count'] > df['Friends_Count'].median()).astype(int)

    features = [
        'Academic_Performance', 'isolated', 'WomenDifferent', 'language',
        'pwi_wellbeing', 'GrowthMindset', 'Wellbeing_Score', 'Manbox5_overall',
        'Masculinity_contrained', 'School_support_engage6', 'School_support_engage', 'bullying'
    ]

    for col in features:
        if col not in df.columns:
            df[col] = 0
        elif df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[features]
    y_academic = df['Academic_Success']
    y_wellbeing = df['Wellbeing_Decline']
    y_peer = df['Positive_Peer_Collab']

    try:
        academic_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(X, y_academic)
        wellbeing_model = RandomForestClassifier(random_state=42).fit(X, y_wellbeing)
        peer_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(X, y_peer)

        df['Academic_Risk'] = academic_model.predict_proba(X)[:, 0]
        df['Wellbeing_Risk'] = wellbeing_model.predict_proba(X)[:, 1]
        df['Peer_Score'] = peer_model.predict_proba(X)[:, 1]
    except Exception as e:
        print(f"Error during model training/prediction: {e}")
        df['Academic_Risk'] = 0.5
        df['Wellbeing_Risk'] = 0.5
        df['Peer_Score'] = 0.5

    print("Analysis complete.")
    return df

# --- Genetic Algorithm for Class Assignment ---
class ClassroomGAProblem(ElementwiseProblem):
    def __init__(self, df, class_size_limit=25, max_bullies_per_class=2, wellbeing_min=None, wellbeing_max=None):
        num_students = len(df)
        num_classes = int(np.ceil(num_students / class_size_limit))
        super().__init__(
            n_var=num_students, n_obj=1, n_constr=0, xl=0, xu=num_classes-1, vtype=int
        )
        self.df = df
        self.num_students = num_students
        self.num_classes = num_classes
        self.class_size_limit = class_size_limit
        self.max_bullies_per_class = max_bullies_per_class
        self.academic_perf = df['Academic_Performance'].values
        self.bullying_scores = df['Bullying_Score'].values if 'Bullying_Score' in df.columns else np.zeros(num_students)
        self.wellbeing_scores = df['Wellbeing_Score'].values if 'Wellbeing_Score' in df.columns else np.zeros(num_students)
        self.wellbeing_min = wellbeing_min
        self.wellbeing_max = wellbeing_max

    def _evaluate(self, x, out, *args, **kwargs):
        # x: array of class assignments for each student
        class_sizes = np.zeros(self.num_classes, dtype=int)
        class_totals = np.zeros(self.num_classes, dtype=float)
        class_bullies = np.zeros(self.num_classes, dtype=int)
        class_wellbeing = [[] for _ in range(self.num_classes)]
        for i, cls in enumerate(x):
            class_sizes[cls] += 1
            class_totals[cls] += self.academic_perf[i]
            if self.bullying_scores[i] > 7:
                class_bullies[cls] += 1
            class_wellbeing[cls].append(self.wellbeing_scores[i])
        # Penalize class size violations
        min_class_size = self.num_students // self.num_classes
        num_larger_classes = self.num_students % self.num_classes
        max_class_size = min_class_size + 1 if num_larger_classes > 0 else min_class_size
        size_penalty = 0
        for j in range(self.num_classes):
            if j < num_larger_classes:
                if class_sizes[j] != max_class_size:
                    size_penalty += abs(class_sizes[j] - max_class_size) * 1000
            else:
                if class_sizes[j] != min_class_size:
                    size_penalty += abs(class_sizes[j] - min_class_size) * 1000
        # Penalize if academic totals differ too much
        total_penalty = 0
        for i in range(self.num_classes):
            for j in range(i+1, self.num_classes):
                diff = abs(class_totals[i] - class_totals[j])
                if diff > 15 * max_class_size:  # similar to JS: allow some difference
                    total_penalty += (diff - 15 * max_class_size) * 10
        # Penalize bullying constraint
        bully_penalty = np.sum(np.maximum(0, class_bullies - self.max_bullies_per_class)) * 1000
        # Penalize wellbeing constraint (optional, if min/max provided)
        wellbeing_penalty = 0
        if self.wellbeing_min is not None or self.wellbeing_max is not None:
            for wlist in class_wellbeing:
                if wlist:
                    avg_well = np.mean(wlist)
                    if self.wellbeing_min is not None and avg_well < self.wellbeing_min:
                        wellbeing_penalty += (self.wellbeing_min - avg_well) * 100
                    if self.wellbeing_max is not None and avg_well > self.wellbeing_max:
                        wellbeing_penalty += (avg_well - self.wellbeing_max) * 100
        out["F"] = [size_penalty + total_penalty + bully_penalty + wellbeing_penalty]

def solve_with_genetic_algorithm(df, class_size_limit=25, max_bullies_per_class=2, wellbeing_min=None, wellbeing_max=None, generations=50, pop_size=100):
    problem = ClassroomGAProblem(df, class_size_limit=class_size_limit, max_bullies_per_class=max_bullies_per_class, wellbeing_min=wellbeing_min, wellbeing_max=wellbeing_max)
    algorithm = GA(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SinglePointCrossover(prob=0.9),
        mutation=BitflipMutation(prob=1.0 / len(df)),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", generations)
    res = minimize(problem, algorithm, termination, seed=1, verbose=False)
    if res.X is not None:
        num_classes = int(np.ceil(len(df) / class_size_limit))
        valid_assignments = np.array([
            x if isinstance(x, (int, np.integer)) and 0 <= x < num_classes else -1
            for x in res.X
        ])
        df["Class_GA"] = valid_assignments
        # Build result structure
        classes = []
        violations = []
        for class_id in range(num_classes):
            students = df[df["Class_GA"] == class_id]
            classes.append({
                "classId": class_id,
                "students": [
                    {
                        "id": row["StudentID"] if "StudentID" in row else row["Student_ID"],
                        "academicScore": row["Academic_Performance"],
                        "wellbeingScore": row["Wellbeing_Score"],
                        "bullyingScore": row["Bullying_Score"]
                    }
                    for _, row in students.iterrows()
                ]
            })
            # Constraint checks
            if len(students) > class_size_limit:
                violations.append(f"Class {class_id} exceeds max size ({len(students)}/{class_size_limit})")
            if "Bullying_Score" in df.columns:
                bullies = students[students["Bullying_Score"] > 7]
                if len(bullies) > max_bullies_per_class:
                    violations.append(f"Class {class_id} has {len(bullies)} bullies (max {max_bullies_per_class})")
        # Metrics
        total_students = len(df)
        avg_academic = df["Academic_Performance"].mean()
        avg_wellbeing = df["Wellbeing_Score"].mean()
        metrics = {
            "totalStudents": total_students,
            "numClasses": num_classes,
            "avgAcademic": avg_academic,
            "avgWellbeing": avg_wellbeing,
            "balanceScore": 1.0,  # Placeholder
            "diversityScore": 1.0,  # Placeholder
            "constraintSatisfaction": 1.0 if not violations else 0.5,
            "processingTime": "N/A"
        }
        return {
            "success": True,
            "metrics": metrics,
            "violations": violations,
            "classes": classes
        }
    else:
        return {
            "success": False,
            "metrics": {},
            "violations": ["No solution found"],
            "classes": []
        }

# --- Flask API ---
app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the frontend

@app.route("/allocate", methods=["POST"])
def allocate():
    try:
        data = request.get_json()
        students = data["students"]
        params = data.get("params", {})
        # Map JS fields to DataFrame columns
        df = pd.DataFrame([{
            "StudentID": s.get("id", s.get("Student_ID")),
            "Academic_Performance": s.get("academicScore", s.get("Academic_Performance")),
            "Wellbeing_Score": s.get("wellbeingScore", s.get("Wellbeing_Score")),
            "Bullying_Score": s.get("bullyingScore", s.get("Bullying_Score"))
        } for s in students])
        # Run GA
        result = solve_with_genetic_algorithm(
            df,
            class_size_limit=params.get("maxClassSize", 25),
            max_bullies_per_class=params.get("maxBulliesPerClass", 2),
            wellbeing_min=params.get("wellbeingMin"),
            wellbeing_max=params.get("wellbeingMax"),
            generations=params.get("generations", 50),
            pop_size=params.get("populationSize", 100)
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        student_count = data.get("studentCount", 1000)
        include_social = data.get("includeSocial", False)
        include_mental_health = data.get("includeMentalHealth", False)

        # Generate synthetic data
        df = generate_synthetic_data(num_students=student_count)

        # Filter columns based on user preferences
        required_columns = ['Student_ID', 'Academic_Performance', 'Wellbeing_Score', 'Bullying_Score']
        optional_columns = {
            "social": ['Friends', 'Disrespect'],
            "mentalHealth": ['K6_Score', 'Anxiety_Level', 'Depression_Level']
        }

        selected_columns = required_columns
        if include_social:
            selected_columns += optional_columns["social"]
        if include_mental_health:
            selected_columns += optional_columns["mentalHealth"]

        df = df[selected_columns] if all(col in df.columns for col in selected_columns) else df[required_columns]

        # Convert DataFrame to JSON format
        headers = df.columns.tolist()
        rows = df.values.tolist()

        return jsonify({
            "success": True,
            "headers": headers,
            "rows": rows
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("Launching Flask API...")
    app.run(port=5001, debug=True)