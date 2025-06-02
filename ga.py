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
        
        # Ensure StudentID is string for consistent lookup
        self.df['StudentID'] = self.df['StudentID'].astype(str)
        
        self.friends_indices = []
        if 'Friends' in df.columns:
            id_list = list(self.df['StudentID']) # Already strings
            id_to_idx = {sid: idx for idx, sid in enumerate(id_list)}
            for friends_str in df['Friends'].fillna(''):
                if isinstance(friends_str, str) and friends_str.strip():
                    # Friend IDs are expected to be strings matching StudentID
                    friend_ids = [fid.strip() for fid in friends_str.split(',') if fid.strip()]
                    self.friends_indices.append([id_to_idx[fid] for fid in friend_ids if fid in id_to_idx])
                else:
                    self.friends_indices.append([])
        else:
            self.friends_indices = [[] for _ in range(num_students)]

        # Build social network graph
        self.social_graph = nx.Graph()
        self.social_graph.add_nodes_from(range(self.num_students))
        for student_idx, friend_indices_list in enumerate(self.friends_indices):
            for friend_idx in friend_indices_list:
                self.social_graph.add_edge(student_idx, friend_idx)

    def _evaluate(self, x, out, *args, **kwargs):
        # x: array of class assignments for each student
        class_sizes = np.zeros(self.num_classes, dtype=int)
        class_totals = np.zeros(self.num_classes, dtype=float)
        class_bullies = np.zeros(self.num_classes, dtype=int)
        class_wellbeing = [[] for _ in range(self.num_classes)]
        
        # Store student indices per class for SNA
        class_student_indices = [[] for _ in range(self.num_classes)]

        for i, cls in enumerate(x):
            class_sizes[cls] += 1
            class_totals[cls] += self.academic_perf[i]
            if self.bullying_scores[i] > 7:
                class_bullies[cls] += 1
            class_wellbeing[cls].append(self.wellbeing_scores[i])
            class_student_indices[cls].append(i) # Store student index 'i' in class 'cls'
        # Penalize class size violations
        min_class_size = self.num_students // self.num_classes
        num_larger_classes = self.num_students % self.num_classes
        max_class_size = min_class_size + 1 if num_larger_classes > 0 else min_class_size
        size_penalty = 0
        for j in range(self.num_classes):  # Added closing parenthesis
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
        # Social penalty: penalize if friends are split across classes
        friends_penalty = 0
        for i, friends in enumerate(self.friends_indices):
            for fidx in friends:
                if x[i] != x[fidx]:
                    friends_penalty += 1  # Each split friend pair adds penalty
        
        # SNA Penalty: Penalize low social cohesion within classes
        social_cohesion_penalty = 0
        SNA_WEIGHT = 50  # Weight for social cohesion penalty, can be tuned
        SNA_ISOLATION_PENALTY = 50 # Penalty for a single student in a class from SNA perspective

        for cls_idx in range(self.num_classes):
            students_in_this_class = class_student_indices[cls_idx]
            if len(students_in_this_class) > 1:
                # Create a subgraph for the current class
                class_subgraph = self.social_graph.subgraph(students_in_this_class)
                # Calculate average clustering coefficient for the class subgraph
                # nx.average_clustering returns 0 if no triangles (e.g., a line graph or isolated nodes)
                avg_clustering = nx.average_clustering(class_subgraph)
                social_cohesion_penalty += (1 - avg_clustering) * SNA_WEIGHT
            elif len(students_in_this_class) == 1:
                # Penalize classes with only one student from a social cohesion perspective
                social_cohesion_penalty += SNA_ISOLATION_PENALTY
                
        out["F"] = [size_penalty + total_penalty + bully_penalty + wellbeing_penalty + friends_penalty * 100 + social_cohesion_penalty]

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
        # Ensure 'Friends' column exists, fill NaN with empty string
        if 'Friends' not in df.columns:
            df['Friends'] = ''
        else:
            df['Friends'] = df['Friends'].fillna('')
        
        # Ensure StudentID is string for the output
        df['StudentID'] = df['StudentID'].astype(str)

        for class_id in range(num_classes):
            students_in_class_df = df[df["Class_GA"] == class_id]
            classes.append({
                "classId": class_id,
                "students": [
                    {
                        "id": row["StudentID"], # Already a string
                        "academicScore": row["Academic_Performance"],
                        "wellbeingScore": row["Wellbeing_Score"],
                        "bullyingScore": row["Bullying_Score"],
                        "friends": row["Friends"] 
                    }
                    for _, row in students_in_class_df.iterrows()
                ]
            })
            # Constraint checks
            if len(students_in_class_df) > class_size_limit:
                violations.append(f"Class {class_id} exceeds max size ({len(students_in_class_df)}/{class_size_limit})")
            if "Bullying_Score" in df.columns:
                bullies = students_in_class_df[students_in_class_df["Bullying_Score"] > 7]
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

@app.route("/", methods=["GET"])
def index():
    """Root endpoint providing API documentation."""
    return jsonify({
        "service": "Classroom Allocation API",
        "version": "1.0",
        "endpoints": {
            "/allocate": "POST - Submit students data and parameters to generate classroom allocations",
            "/generate": "POST - (Deprecated) Synthetic data generation moved to frontend"
        },
        "status": "running"
    })

@app.route("/allocate", methods=["POST"])
def allocate():
    try:
        data = request.get_json()
        students_data = data["students"]
        params = data.get("params", {})

        # Filter out students with missing 'id' before creating DataFrame
        valid_students_data = [s for s in students_data if s.get("id") is not None]
        if not valid_students_data:
            return jsonify({"success": False, "error": "No valid student data received (all students missing ID)."}), 400
        
        # Map JS fields to DataFrame columns, ensuring StudentID is primary
        df = pd.DataFrame([{
            "StudentID": str(s.get("id")), # Ensure StudentID is a string and is the primary ID field
            "Academic_Performance": s.get("academicScore", s.get("Academic_Performance")),
            "Wellbeing_Score": s.get("wellbeingScore", s.get("Wellbeing_Score")),
            "Bullying_Score": s.get("bullyingScore", s.get("Bullying_Score")),
            "Friends": s.get("friends", s.get("Friends"))
        } for s in valid_students_data])
        
        if df.empty:
            return jsonify({"success": False, "error": "DataFrame is empty after processing student data."}), 400
            
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
    return jsonify({"success": False, "error": "Synthetic data generation has been moved to the frontend."}), 400

if __name__ == "__main__":
    print("Launching Flask API...")
    app.run(port=5001, debug=True)