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
                required_cols = ['StudentID', 'Academic_Performance', 'Friends', 'bullying']
                if all(col in df.columns for col in required_cols):
                    print("Data loaded successfully.")
                    return df
                else:
                    print("Required columns missing in CSV, regenerating...")
            else:
                print(f"CSV found but has {len(df)} rows, expected {num_students}. Regenerating...")
        except Exception as e:
            print(f"Error loading CSV: {e}. Regenerating...")

    print(f"Generating {num_students} new synthetic student records...")
    LIKERT_SCALE_1_7 = list(range(1, 8))
    K6_SCALE_1_5 = list(range(1, 6))
    LANGUAGE_SCALE = [0, 1]
    PWI_SCALE = list(range(0, 11))

    student_ids = [f"S{i:04d}" for i in range(1, num_students + 1)]
    data = []

    for student_id in student_ids:
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

        # 5% chance for extreme bullying
        if random.random() < 0.05:
            bullying = random.choice([1, 7])
        else:
            bullying = random.choice(LIKERT_SCALE_1_7)

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
        }
        data.append(student_data)

    df = pd.DataFrame(data)

    # Derived Fields
    df['Manbox5_overall'] = df[[f"Manbox5_{i}" for i in range(1, 6)]].mean(axis=1)
    df['Masculinity_contrained'] = df[['Soft', 'WomenDifferent', 'Nerds', 'MenBetterSTEM']].mean(axis=1)
    df['GrowthMindset'] = ((8.0 - df['Intelligence1']) + (8.0 - df['Intelligence2'])) / 2.0
    df['k6_overall'] = df[[f"k6_{i}" for i in range(1, 7)]].sum(axis=1)
    df['School_support_engage6'] = ((8.0 - df['isolated']) + (8.0 - df['opinion']) + df['criticises'] + df['comfortable'] + df['bullying'] + df['future']) / 6.0
    df['School_support_engage'] = df[['criticises', 'comfortable', 'bullying', 'future']].mean(axis=1)

    df.to_csv(filename, index=False)
    print(f"Synthetic data saved to {filename}")
    return df

# --- Predictive Analytics ---
def run_analysis(df):
    """Performs prediction on the dataframe (no clustering)."""
    print("Running predictive analytics...")
    df['Academic_Success'] = (df['Academic_Performance'] > df['Academic_Performance'].quantile(0.75)).astype(int)
    df['Wellbeing_Decline'] = (df['k6_overall'] > df['k6_overall'].quantile(0.75)).astype(int)
    df['Friends_Count'] = df['Friends'].fillna('').apply(lambda x: len(x.split(', ')) if x else 0)
    df['Positive_Peer_Collab'] = (df['Friends_Count'] > df['Friends_Count'].median()).astype(int)

    features = [
        'Academic_Performance', 'isolated', 'WomenDifferent', 'language',
        'pwi_wellbeing', 'GrowthMindset', 'k6_overall', 'Manbox5_overall',
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
    def __init__(self, df, class_size_limit=25, max_allowed_total_diff=200):
        num_students = len(df)
        num_classes = int(np.ceil(num_students / class_size_limit))
        super().__init__(
            n_var=num_students, n_obj=1, n_constr=0, xl=0, xu=num_classes-1, vtype=int
        )
        self.df = df
        self.num_students = num_students
        self.num_classes = num_classes
        self.class_size_limit = class_size_limit
        self.max_allowed_total_diff = max_allowed_total_diff
        self.academic_perf = df['Academic_Performance'].values

    def _evaluate(self, x, out, *args, **kwargs):
        # x: array of class assignments for each student
        class_sizes = np.zeros(self.num_classes, dtype=int)
        class_totals = np.zeros(self.num_classes, dtype=int)
        for i, cls in enumerate(x):
            class_sizes[cls] += 1
            class_totals[cls] += self.academic_perf[i]
        # Penalize if class sizes are not within allowed bounds
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
                if diff > self.max_allowed_total_diff:
                    total_penalty += (diff - self.max_allowed_total_diff) * 10
        # Objective: minimize penalties
        out["F"] = [size_penalty + total_penalty]

def solve_with_genetic_algorithm(df):
    print("Solving with genetic algorithm...")
    class_size_limit = 25
    problem = ClassroomGAProblem(df, class_size_limit=class_size_limit)
    algorithm = GA(
        pop_size=100,
        sampling=IntegerRandomSampling(),
        crossover=SinglePointCrossover(prob=0.9),
        mutation=BitflipMutation(prob=1.0 / len(df)),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", 50)
    res = minimize(problem, algorithm, termination, seed=1, verbose=False)
    if res.X is not None:
        print("GA solution found!")
        # Ensure all class assignments are valid integers in [0, num_classes-1]
        num_classes = int(np.ceil(len(df) / class_size_limit))
        valid_assignments = np.array([
            x if isinstance(x, (int, np.integer)) and 0 <= x < num_classes else -1
            for x in res.X
        ])
        df["Class_GA"] = valid_assignments

        # --- Store output classes with students' IDs in a JSON file ---
        class_dict = {}
        for class_id in sorted(set(valid_assignments)):
            if class_id == -1:
                continue  # skip invalid assignments
            student_ids = df.loc[df["Class_GA"] == class_id, "StudentID"].tolist()
            class_dict[int(class_id)] = student_ids
        output_json_path = "class_assignments.json"
        with open(output_json_path, "w") as f:
            json.dump(class_dict, f, indent=2)
        print(f"Class assignments saved to {output_json_path}")
        # -------------------------------------------------------------
    else:
        print("No GA solution found.")
        df["Class_GA"] = -1
    return df

# --- Visualization Functions ---
def plot_network(df_class, student_id_col='StudentID', friend_col='Friends', color_metric='Academic_Risk'):
    """Generates a NetworkX graph visualization for a class."""
    if df_class.empty:
        return None  # Return None if no data

    G = nx.Graph()
    all_students_in_class = df_class[student_id_col].tolist()
    G.add_nodes_from(all_students_in_class)

    # Map student IDs to their metric value for coloring
    metric_values = df_class.set_index(student_id_col)[color_metric]
    min_val, max_val = metric_values.min(), metric_values.max()
    cmap = cm.viridis  # Choose colormap

    node_colors = {}
    for node in G.nodes():
        value = metric_values.get(node, 0)  # Default to 0 if not found
        # Normalize value to 0-1 for colormap
        normalized_value = (value - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0.5
        node_colors[node] = cmap(normalized_value)

    # Add edges based on the 'Friends' column (within the class)
    for index, row in df_class.iterrows():
        student = row[student_id_col]
        friends_str = row[friend_col]
        if pd.isna(friends_str) or not friends_str:
            continue
        friends_list = [f.strip() for f in friends_str.split(',')]
        for friend in friends_list:
            if friend in all_students_in_class:  # Add edge only if the friend is in the same class
                G.add_edge(student, friend)

    fig, ax = plt.subplots(figsize=(10, 8))  # Increase figure size
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)  # Adjust k for spacing

    nx.draw(
        G, pos, ax=ax,
        node_size=50,  # Smaller nodes
        width=0.5,  # Thinner edges
        with_labels=False,  # No labels to avoid clutter
        node_color=[node_colors.get(node, cmap(0.5)) for node in G.nodes()]  # Apply colors
    )

    ax.set_title(f"Friendship Network within Class (Colored by {color_metric})")

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
    sm.set_array([])  # Required for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label(color_metric)

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)  # Close the figure to free memory
    return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%"/>'

def plot_histogram(df_class, metric, class_id):
    """Generates a histogram for a given metric within a class."""
    if df_class.empty or metric not in df_class.columns:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    df_class[metric].plot(kind='hist', ax=ax, bins=10, alpha=0.7)  # Use 10 bins
    ax.set_title(f"{metric} Distribution in Class {class_id}")
    ax.set_xlabel(metric)
    ax.set_ylabel("Frequency")
    plt.tight_layout()

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)  # Close the figure to free memory
    # Wrap in <img> tag for Gradio HTML component
    return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%"/>'

# Example: To print the DLL path as a string (for debugging or info)
print(r"C:\Program Files\AHSDK\bin\ahscript.dll")

# Process the data and solve with genetic algorithm
df_processed = solve_with_genetic_algorithm(run_analysis(generate_synthetic_data()))

# Extract unique class IDs for dropdown
all_classes = sorted(df_processed['Class_GA'].unique())

def update_visualizations(selected_class_id, color_metric):
    """Updates the Gradio interface based on user selections."""
    class_col = 'Class_GA'
    df_selected_class = df_processed[df_processed[class_col] == selected_class_id]
    overview = df_processed.groupby(class_col).agg(
        Class_Size=('StudentID', 'count'),
        Avg_Academic_Perf=('Academic_Performance', 'mean'),
        Avg_Academic_Risk=('Academic_Risk', 'mean'),
        Avg_Wellbeing_Risk=('Wellbeing_Risk', 'mean'),
        Avg_Peer_Score=('Peer_Score', 'mean'),
        Avg_Friends_Count=('Friends_Count', 'mean')
    ).reset_index()
    student_table = df_selected_class[[
        'StudentID', 'Academic_Performance', 'Academic_Risk',
        'Wellbeing_Risk', 'Peer_Score', 'Friends_Count'
    ]]
    network_plot_html = plot_network(df_selected_class, color_metric=color_metric)
    academic_hist_html = plot_histogram(df_selected_class, 'Academic_Risk', selected_class_id)
    wellbeing_hist_html = plot_histogram(df_selected_class, 'Wellbeing_Risk', selected_class_id)
    return overview, student_table, network_plot_html, academic_hist_html, wellbeing_hist_html

with gr.Blocks(theme=gr.themes.Soft(), title="Classroom Allocation Visualizer") as demo:
    gr.Markdown("# AI-Powered Classroom Allocation Visualization")
    gr.Markdown("Explore the results of predictive modeling and classroom allocation using a genetic algorithm. Select a class ID to view details.")

    with gr.Row():
        class_id_dd = gr.Dropdown(
            label="Select Class ID",
            choices=all_classes,
            value=all_classes[0] if all_classes else None,
            interactive=True
        )
        color_metric_dd = gr.Dropdown(
            label="Color Network By",
            choices=['Academic_Risk', 'Wellbeing_Risk', 'Peer_Score', 'Academic_Performance', 'k6_overall', 'Friends_Count'],
            value='Academic_Risk',
            interactive=True
        )

    gr.Markdown("## Class Overview")
    overview_table = gr.DataFrame(label="Summary statistics for all classes")

    gr.Markdown("## Selected Class Details")
    with gr.Row():
        with gr.Column(scale=2):
            student_table = gr.DataFrame(label="Students in Selected Class")
        with gr.Column(scale=3):
            gr.Markdown("### Friendship Network")
            network_plot_html = gr.HTML(label="Class Network Graph")

    gr.Markdown("## Metric Distributions for Selected Class")
    with gr.Row():
        academic_hist_html = gr.HTML(label="Academic Risk Distribution")
        wellbeing_hist_html = gr.HTML(label="Wellbeing Risk Distribution")

    inputs = [class_id_dd, color_metric_dd]
    outputs = [overview_table, student_table, network_plot_html, academic_hist_html, wellbeing_hist_html]

    class_id_dd.change(update_visualizations, inputs=inputs, outputs=outputs)
    color_metric_dd.change(update_visualizations, inputs=inputs, outputs=outputs)
    demo.load(update_visualizations, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch(share=False)
    print("Gradio App launched.")