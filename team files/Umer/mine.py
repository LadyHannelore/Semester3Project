# --- Imports ---
import os
import random
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import gradio as gr
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.spx import SinglePointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
NUM_STUDENTS = 1000
NUM_CLASSES = 40
SYNTHETIC_DATA_CSV = "synthetic_student_data_1000.csv"
GENERATIONS = 50
POPULATION_SIZE = 100

# --- Data Generation ---
def generate_synthetic_data(filename=SYNTHETIC_DATA_CSV, num_students=NUM_STUDENTS):
    """Generates synthetic student data if the CSV doesn't exist or is invalid."""
    if os.path.exists(filename):
        print(f"Loading existing data from {filename}")
        try:
            df = pd.read_csv(filename)
            required_cols = [
                'StudentID', 'Academic_Performance', 'Friends', 'criticises',
                'k6_overall', 'School_support_engage', 'language'
            ]
            if len(df) == num_students and all(col in df.columns for col in required_cols):
                print("Data loaded successfully.")
                return df
            else:
                print(f"CSV found but invalid. Regenerating...")
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
        academic_performance = max(0, min(100, round(np.random.normal(70, 15))))
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
            "bullying": random.choice(LIKERT_SCALE_1_7),
            **{f"Manbox5_{i}": random.choice(LIKERT_SCALE_1_7) for i in range(1, 6)},
            **{f"k6_{i}": random.choice(K6_SCALE_1_5) for i in range(1, 7)},
        }
        data.append(student_data)

    df = pd.DataFrame(data)

    df['Manbox5_overall'] = df[[f"Manbox5_{i}" for i in range(1, 6)]].mean(axis=1)
    df['Masculinity_contrained'] = df[['Soft', 'WomenDifferent', 'Nerds', 'MenBetterSTEM']].mean(axis=1)
    df['GrowthMindset'] = ((8.0 - df['Intelligence1']) + (8.0 - df['Intelligence2'])) / 2.0
    df['k6_overall'] = df[[f"k6_{i}" for i in range(1, 7)]].sum(axis=1)
    df['School_support_engage'] = (df['comfortable'] + df['future'] + (8.0 - df['isolated']) + (8.0 - df['opinion'])) / 4.0

    all_student_ids_set = set(student_ids)
    df['Friends'] = df['StudentID'].apply(
        lambda x: ", ".join(random.sample(list(all_student_ids_set - {x}), k=random.randint(0, min(7, num_students - 1))))
    )
    df['Friends_Count'] = df['Friends'].fillna('').apply(lambda x: len([f for f in x.split(',') if f.strip()]))

    df.to_csv(filename, index=False)
    print(f"Synthetic data saved to {filename}")
    return df

# --- Preprocessing ---
def load_and_preprocess_data(filename=SYNTHETIC_DATA_CSV, num_students=NUM_STUDENTS):
    """Loads data, performs basic preprocessing, and builds the social graph."""
    df = generate_synthetic_data(filename, num_students)
    df['Academic_Performance_Norm'] = df['Academic_Performance'] / 100.0

    G = nx.Graph()
    student_id_map = {sid: i for i, sid in enumerate(df['StudentID'])}
    id_student_map = {i: sid for sid, i in student_id_map.items()}

    for _, row in df.iterrows():
        student_id = row['StudentID']
        G.add_node(student_id_map[student_id], **row.to_dict())
        friends_str = row['Friends']
        if pd.notna(friends_str) and friends_str.strip():
            friends_list = [f.strip() for f in friends_str.split(',') if f.strip()]
            for friend_id in friends_list:
                if friend_id in student_id_map:
                    if student_id_map[student_id] != student_id_map[friend_id]:
                        G.add_edge(student_id_map[student_id], student_id_map[friend_id], type='friend')

    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return df, G, student_id_map, id_student_map

# --- Optimization Engine ---
class ClassroomAllocationProblem(ElementwiseProblem):
    """Defines the multi-objective optimization problem for classroom allocation."""
    def __init__(self, n_students, n_classes, df, G, student_id_map):
        super().__init__(
            n_var=n_students, n_obj=3, n_constr=0, xl=0, xu=n_classes - 1, vtype=int
        )
        self.n_students = n_students
        self.n_classes = n_classes
        self.df = df
        self.G = G
        self.student_id_map = student_id_map
        self.academic_perf = df['Academic_Performance_Norm'].values
        self.k6_scores = df['k6_overall'].values
        self.criticises_scores = df['criticises'].values
        self.friend_pairs = list(G.edges())

    def _evaluate(self, x, out, *args, **kwargs):
        class_academic_perf = [[] for _ in range(self.n_classes)]
        class_k6_scores = [[] for _ in range(self.n_classes)]
        class_criticises_scores = [[] for _ in range(self.n_classes)]
        students_in_class = [[] for _ in range(self.n_classes)]

        for student_idx, class_idx in enumerate(x):
            if 0 <= class_idx < self.n_classes:
                class_academic_perf[class_idx].append(self.academic_perf[student_idx])
                class_k6_scores[class_idx].append(self.k6_scores[student_idx])
                class_criticises_scores[class_idx].append(self.criticises_scores[student_idx])
                students_in_class[class_idx].append(student_idx)
            else:
                print(f"Warning: Invalid class index {class_idx} for student {student_idx}")
                out["F"] = [np.inf, np.inf, np.inf]
                return

        avg_perf_per_class = [np.mean(p) if p else 0 for p in class_academic_perf]
        f1 = np.var(avg_perf_per_class)

        avg_k6_per_class = [np.mean(s) if s else 0 for s in class_k6_scores]
        f2 = np.var(avg_k6_per_class)

        retained_friendships = 0
        for u, v in self.friend_pairs:
            if x[u] == x[v]:
                retained_friendships += 1
        f3 = -retained_friendships

        out["F"] = [f1, f2, f3]

def run_optimization(n_students, n_classes, df, G, student_id_map, generations, pop_size):
    """Runs the NSGA-II optimization."""
    problem = ClassroomAllocationProblem(n_students, n_classes, df, G, student_id_map)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SinglePointCrossover(prob=0.9),
        mutation=BitflipMutation(prob=1.0 / n_students),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", generations)
    print(f"Starting optimization with {generations} generations, {pop_size} population size...")
    res = minimize(problem, algorithm, termination, seed=1, verbose=True)
    if res.F is None or len(res.F) == 0:
        print("Optimization failed to find solutions.")
        return None, None
    return res.X, res.F

# --- Visualization and UI ---
def plot_pareto_front(pareto_f):
    """Generates a 3D scatter plot of the Pareto front."""
    if pareto_f is None or pareto_f.shape[1] != 3:
        return go.Figure()

    f1 = pareto_f[:, 0]
    f2 = pareto_f[:, 1]
    f3 = -pareto_f[:, 2]

    fig = go.Figure(data=[go.Scatter3d(
        x=f1, y=f2, z=f3,
        mode='markers',
        marker=dict(
            size=5,
            color=f3,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Retained Friendships')
        ),
        hovertemplate=
        '<b>Academic Var</b>: %{x:.3f}<br>' +
        '<b>Wellbeing Var</b>: %{y:.3f}<br>' +
        '<b>Friendships</b>: %{z}<br>' +
        '<extra></extra>'
    )])

    fig.update_layout(
        title='Pareto Front: Trade-offs between Objectives',
        scene=dict(
            xaxis_title='Academic Variance (Minimize)',
            yaxis_title='Wellbeing (K6) Variance (Minimize)',
            zaxis_title='Retained Friendships (Maximize)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def select_solution_from_weights(pareto_x, pareto_f, w_academic, w_wellbeing, w_social):
    """Selects a solution from the Pareto front based on normalized weights."""
    if pareto_f is None or len(pareto_f) == 0:
        return None, "No solutions found."

    min_f = np.min(pareto_f, axis=0)
    max_f = np.max(pareto_f, axis=0)
    range_f = max_f - min_f
    range_f[range_f == 0] = 1

    normalized_f = (pareto_f - min_f) / range_f

    total_weight = w_academic + w_wellbeing + w_social
    if total_weight == 0:
        total_weight = 1
    weights = np.array([w_academic, w_wellbeing, w_social]) / total_weight

    weighted_sums = np.sum(normalized_f * weights, axis=1)
    best_idx = np.argmin(weighted_sums)

    selected_allocation = pareto_x[best_idx]
    selected_objectives = pareto_f[best_idx]

    summary = (
        f"Selected Solution (Index {best_idx}):\n"
        f"- Academic Perf. Variance: {selected_objectives[0]:.4f}\n"
        f"- Wellbeing (K6) Variance: {selected_objectives[1]:.4f}\n"
        f"- Retained Friendships: {-selected_objectives[2]}"
    )
    return selected_allocation, summary

def format_allocation_df(allocation, df, id_student_map):
    """Formats the selected allocation into a readable DataFrame."""
    if allocation is None:
        return pd.DataFrame()

    alloc_dict = []
    for student_idx, class_id in enumerate(allocation):
        student_sid = id_student_map.get(student_idx, f"UnknownIdx_{student_idx}")
        student_info = df[df['StudentID'] == student_sid].iloc[0]
        alloc_dict.append({
            'StudentID': student_sid,
            'Assigned_Class': class_id,
            'Academic_Performance': student_info['Academic_Performance'],
            'K6_Overall': student_info['k6_overall'],
            'Criticises_Score': student_info['criticises'],
            'Friends_Count': student_info['Friends_Count']
        })
    return pd.DataFrame(alloc_dict)

def generate_classroom_allocation(num_classes_ui, w_academic, w_wellbeing, w_social, progress=gr.Progress(track_tqdm=True)):
    """Main function called by Gradio button."""
    progress(0, desc="Loading and Preprocessing Data...")
    try:
        num_classes_int = int(num_classes_ui)
        if num_classes_int <= 0:
            raise ValueError("Number of classes must be positive.")

        df, G, student_id_map, id_student_map = load_and_preprocess_data(num_students=NUM_STUDENTS)
        n_students_actual = len(df)

        progress(0.2, desc=f"Running NSGA-II Optimization ({GENERATIONS} Gens)...")
        pareto_x, pareto_f = run_optimization(
            n_students_actual, num_classes_int, df, G, student_id_map, GENERATIONS, POPULATION_SIZE
        )
        if pareto_x is None:
            return None, "Optimization failed.", pd.DataFrame(), "No results."

        progress(0.8, desc="Processing Results...")
        pareto_plot = plot_pareto_front(pareto_f)
        selected_allocation, selection_summary = select_solution_from_weights(
            pareto_x, pareto_f, w_academic, w_wellbeing, w_social
        )
        allocation_df = format_allocation_df(selected_allocation, df, id_student_map)

        if selected_allocation is not None:
            class_summary = "\n\nClass Summaries:\n"
            class_acad = [[] for _ in range(num_classes_int)]
            class_k6 = [[] for _ in range(num_classes_int)]
            class_friends = [0] * num_classes_int
            class_students = [[] for _ in range(num_classes_int)]
            friend_pairs = list(G.edges())

            for idx, cls in enumerate(selected_allocation):
                class_acad[cls].append(df.iloc[idx]['Academic_Performance'])
                class_k6[cls].append(df.iloc[idx]['k6_overall'])
                class_students[cls].append(id_student_map[idx])

            for u, v in friend_pairs:
                if u < len(selected_allocation) and v < len(selected_allocation):
                    if selected_allocation[u] == selected_allocation[v]:
                        class_friends[selected_allocation[u]] += 1

            for i in range(num_classes_int):
                avg_acad = np.mean(class_acad[i]) if class_acad[i] else 0
                avg_k6 = np.mean(class_k6[i]) if class_k6[i] else 0
                num_stud = len(class_students[i])
                friends_in_class = class_friends[i]
                class_summary += (
                    f" Class {i}: {num_stud} students, "
                    f"Avg Perf: {avg_acad:.1f}, Avg K6: {avg_k6:.2f}, "
                    f"Internal Friend Pairs: {friends_in_class}\n"
                )
            selection_summary += class_summary

        progress(1.0, desc="Done!")
        return pareto_plot, selection_summary, allocation_df, "Allocation generated successfully."

    except Exception as e:
        print(f"Error during allocation generation: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {e}", pd.DataFrame(), f"Failed: {e}"

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ClassForge: AI Powered Customizable Classroom Allocation
        Configure parameters and run the optimizer to generate classroom assignments based on academic performance,
        student well-being (K6 scores), and social network connections (friendship retention).
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            num_classes_input = gr.Number(label="Number of Classes", value=NUM_CLASSES, minimum=1, step=1)
            gr.Markdown("### Set Allocation Priorities (Weights)")
            weight_academic = gr.Slider(label="Academic Equity", minimum=0.0, maximum=1.0, value=0.4, step=0.1)
            weight_wellbeing = gr.Slider(label="Well-being Balance (K6)", minimum=0.0, maximum=1.0, value=0.3, step=0.1)
            weight_social = gr.Slider(label="Social Cohesion (Friendships)", minimum=0.0, maximum=1.0, value=0.3, step=0.1)
            run_button = gr.Button("Generate Allocation", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=3):
            gr.Markdown("### Optimization Results")
            pareto_plot_output = gr.Plot(label="Pareto Front (Trade-offs)")
            selected_summary_output = gr.Textbox(label="Selected Solution Summary", lines=8, interactive=False)
            allocation_table_output = gr.DataFrame(label="Selected Classroom Allocation", wrap=True)

    run_button.click(
        fn=generate_classroom_allocation,
        inputs=[num_classes_input, weight_academic, weight_wellbeing, weight_social],
        outputs=[pareto_plot_output, selected_summary_output, allocation_table_output, status_output]
    )

if __name__ == "__main__":
    print("Checking for synthetic data...")
    _ = generate_synthetic_data(SYNTHETIC_DATA_CSV, NUM_STUDENTS)
    print("Launching Gradio App...")
    demo.launch(debug=True)