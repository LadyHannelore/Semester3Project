# Install prerequisites:
# pip install pyomo pandas numpy networkx matplotlib scikit-learn xgboost gradio
# AND install a solver like CBC or GLPK (see instructions above, conda recommended)

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import os
import time
# --- Pyomo Imports ---
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
# --- End Pyomo Imports ---

# --- Configuration & Setup ---
# ... (Keep existing configuration: SYNTHETIC_DATA_CSV, NUM_STUDENTS, etc.) ...
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=DeprecationWarning) # Pyomo can raise these
plt.style.use('seaborn-v0_8-whitegrid')

SYNTHETIC_DATA_CSV = "synthetic_student_data.csv"
NUM_STUDENTS = 100 # Reduced for faster testing
CLASS_SIZE_TARGET = 25
N_CLASSES = max(1, round(NUM_STUDENTS / CLASS_SIZE_TARGET))

# --- Hard Constraints (for CP/Pyomo) ---
CLASS_SIZE_TOLERANCE = 3
AVG_CLASS_SIZE = NUM_STUDENTS // N_CLASSES
CLASS_SIZE_MIN = max(1, AVG_CLASS_SIZE - CLASS_SIZE_TOLERANCE)
CLASS_SIZE_MAX = AVG_CLASS_SIZE + CLASS_SIZE_TOLERANCE + (NUM_STUDENTS % N_CLASSES > 0)

# Define thresholds and parameters for Objectives/GA
BULLY_CRITICISES_THRESHOLD = 6
VULNERABLE_WELLBEING_QUANTILE = 0.8

# Genetic Algorithm Parameters
GA_POP_SIZE = 100
GA_NUM_GENERATIONS = 50
GA_ELITISM_RATE = 0.05
GA_MUTATION_RATE_LOW = 0.02
GA_MUTATION_RATE_HIGH = 0.05
GA_TOURNAMENT_SIZE = 5
# CP_SEED_COUNT = 5 # Now Pyomo Seed Count
PYOMO_SEED_COUNT = 3 # Adjust as needed, Pyomo might be slower than CP-SAT
PYOMO_SOLVER = 'cbc' # Change to 'glpk' if you installed that, or provide path if needed

# Adaptive Mutation Threshold
FITNESS_VARIANCE_THRESHOLD = 0.0005
FITNESS_IMPROVEMENT_THRESHOLD = 0.001
IMPROVEMENT_CHECK_GENERATIONS = 10

# Weights for the *scalar* fitness function
FITNESS_WEIGHTS = {
    'academic_equity': 2.0,
    'wellbeing_balance': 1.5,
    'social_cohesion': 3.0,
}

# Constraint Violation Penalty
CONSTRAINT_PENALTY = 10000.0

# --- Data Generation (Keep as is) ---
def generate_synthetic_data(filename=SYNTHETIC_DATA_CSV, num_students=NUM_STUDENTS):
    # ... (no changes needed in this function) ...
    """Generates synthetic student data if the CSV doesn't exist or is invalid."""
    if os.path.exists(filename):
        print(f"Loading existing data from {filename}")
        try:
            df = pd.read_csv(filename)
            required_cols = ['StudentID', 'Academic_Performance', 'Friends', 'criticises', 'k6_overall', 'School_support_engage', 'language']
            # Ensure loaded data matches NUM_STUDENTS if reloading
            if len(df) == num_students and all(col in df.columns for col in required_cols):
                print("Data loaded successfully.")
                return df
            else:
                print(f"CSV found but invalid (size {len(df)} != {num_students} or missing columns), regenerating...")
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
            "StudentID": student_id, "Academic_Performance": academic_performance,
            "isolated": random.choice(LIKERT_SCALE_1_7), "WomenDifferent": random.choice(LIKERT_SCALE_1_7),
            "language": random.choices(LANGUAGE_SCALE, weights=[0.8, 0.2], k=1)[0],
            "COVID": random.choice(LIKERT_SCALE_1_7), "criticises": random.choice(LIKERT_SCALE_1_7),
            "MenBetterSTEM": random.choice(LIKERT_SCALE_1_7), "pwi_wellbeing": random.choice(PWI_SCALE),
            "Intelligence1": random.choice(LIKERT_SCALE_1_7), "Intelligence2": random.choice(LIKERT_SCALE_1_7),
            "Soft": random.choice(LIKERT_SCALE_1_7), "opinion": random.choice(LIKERT_SCALE_1_7),
            "Nerds": random.choice(LIKERT_SCALE_1_7), "comfortable": random.choice(LIKERT_SCALE_1_7),
            "future": random.choice(LIKERT_SCALE_1_7), "bullying": random.choice(LIKERT_SCALE_1_7),
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
    df['Friends'] = df['StudentID'].apply(
        lambda x: ", ".join(random.sample([pid for pid in student_ids if pid != x], k=random.randint(0, min(7, num_students - 1))))
    )
    df['Friends_Count'] = df['Friends'].fillna('').apply(lambda x: len([f for f in x.split(',') if f.strip()]))
    df.to_csv(filename, index=False)
    print(f"Synthetic data saved to {filename}")
    return df

# --- Predictive Analysis (Keep as is) ---
def run_predictive_analysis(df):
    # ... (no changes needed in this function) ...
    """Runs predictive models to get risk/score indicators and flags."""
    print("Running predictive analytics for risk scores...")
    raw_cols_needed = [
        'Academic_Performance', 'isolated', 'WomenDifferent', 'language',
        'pwi_wellbeing', 'Intelligence1', 'Intelligence2', 'Soft', 'opinion',
        'Nerds', 'MenBetterSTEM', 'comfortable', 'future', 'bullying', 'criticises',
    ] + [f"Manbox5_{i}" for i in range(1, 6)] + [f"k6_{i}" for i in range(1, 7)]
    for col in raw_cols_needed:
        if col not in df.columns: df[col] = 0
        elif df[col].isnull().any():
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
            except Exception: df[col] = df[col].fillna(0)
    df['Manbox5_overall'] = df[[f"Manbox5_{i}" for i in range(1, 6)]].mean(axis=1)
    df['Masculinity_contrained'] = df[['Soft', 'WomenDifferent', 'Nerds', 'MenBetterSTEM']].mean(axis=1)
    df['GrowthMindset'] = ((8.0 - df['Intelligence1']) + (8.0 - df['Intelligence2'])) / 2.0
    df['k6_overall'] = df[[f"k6_{i}" for i in range(1, 7)]].sum(axis=1)
    df['School_support_engage'] = (df['comfortable'] + df['future'] + (8.0 - df['isolated']) + (8.0 - df['opinion'])) / 4.0
    df['Friends_Count'] = df['Friends'].fillna('').apply(lambda x: len([f for f in x.split(',') if f.strip()]))
    df['Academic_Success'] = (df['Academic_Performance'] > df['Academic_Performance'].quantile(0.75)).astype(int)
    df['Wellbeing_Decline'] = (df['k6_overall'] > df['k6_overall'].quantile(0.75)).astype(int)
    df['Positive_Peer_Collab'] = (df['Friends_Count'] > df['Friends_Count'].median()).astype(int)
    features_for_prediction = [
        'Academic_Performance', 'isolated', 'WomenDifferent', 'language',
        'pwi_wellbeing', 'GrowthMindset', 'k6_overall', 'Manbox5_overall',
        'Masculinity_contrained', 'School_support_engage', 'Friends_Count'
    ]
    X = df[features_for_prediction].copy()
    for col in X.columns:
        if X[col].isnull().any(): X[col] = X[col].fillna(X[col].median())
        if np.isinf(X[col]).any():
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = X[col].fillna(X[col].median())
    y_academic = df['Academic_Success']
    y_wellbeing = df['Wellbeing_Decline']
    y_peer = df['Positive_Peer_Collab']
    try:
        if len(y_academic.unique()) > 1:
            academic_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(X, y_academic)
            df['Academic_Risk'] = 1 - academic_model.predict_proba(X)[:, 1]
        else: df['Academic_Risk'] = 1 - y_academic.iloc[0]
        if len(y_wellbeing.unique()) > 1:
            wellbeing_model = RandomForestClassifier(random_state=42).fit(X, y_wellbeing)
            df['Wellbeing_Risk'] = wellbeing_model.predict_proba(X)[:, 1]
        else: df['Wellbeing_Risk'] = y_wellbeing.iloc[0]
        if len(y_peer.unique()) > 1:
            peer_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(X, y_peer)
            df['Peer_Score'] = peer_model.predict_proba(X)[:, 1]
        else: df['Peer_Score'] = y_peer.iloc[0]
    except Exception as e:
        print(f"Error during model training/prediction: {e}")
        df['Academic_Risk'], df['Wellbeing_Risk'], df['Peer_Score'] = 0.5, 0.5, 0.5

    # Identify Bully/Vulnerable Students (for CP/Pyomo constraints)
    df['Is_Bully'] = (df['criticises'] >= BULLY_CRITICISES_THRESHOLD).astype(int)
    if df['Wellbeing_Risk'].nunique() > 1:
        wellbeing_risk_threshold = df['Wellbeing_Risk'].quantile(VULNERABLE_WELLBEING_QUANTILE)
    else: wellbeing_risk_threshold = df['Wellbeing_Risk'].iloc[0]
    df['Is_Vulnerable'] = (df['Wellbeing_Risk'] >= wellbeing_risk_threshold).astype(int)
    df['Is_Supportive'] = (df['School_support_engage'] >= df['School_support_engage'].quantile(0.8) if df['School_support_engage'].nunique() > 1 else df['School_support_engage'].iloc[0] >= df['School_support_engage'].iloc[0]).astype(int)
    print("Predictive analysis complete.")
    return df

# --- Pyomo Feasibility Helper ---

def generate_pyomo_feasible_solution(df, num_classes, must_separate_pairs, class_min, class_max, solver_name='cbc'):
    """Generates one feasible allocation using Pyomo and a specified solver."""
    print(f"\nAttempting to generate a feasible solution using Pyomo (Solver: {solver_name})...")
    print(f"(Min Size: {class_min}, Max Size: {class_max})")
    start_pyomo_time = time.time()

    model = pyo.ConcreteModel(name="Student_Allocation")

    # --- Sets ---
    student_ids_list = df['StudentID'].tolist()
    model.STUDENTS = pyo.Set(initialize=student_ids_list)
    model.CLASSES = pyo.RangeSet(0, num_classes - 1) # Classes indexed 0 to N-1

    # Filter must_separate_pairs to only include students present in the dataframe
    valid_student_set = set(student_ids_list)
    filtered_separate_pairs = [
        pair for pair in must_separate_pairs
        if pair[0] in valid_student_set and pair[1] in valid_student_set
    ]
    model.MUST_SEPARATE_PAIRS = pyo.Set(initialize=filtered_separate_pairs, dimen=2)
    if len(filtered_separate_pairs) != len(must_separate_pairs):
         print(f"Warning: Filtered separation pairs from {len(must_separate_pairs)} to {len(filtered_separate_pairs)} due to missing students.")
    print(f"Using {len(filtered_separate_pairs)} separation constraints.")

    # --- Variables ---
    # x[student_id, class_idx] = 1 if student is in class, 0 otherwise
    model.x = pyo.Var(model.STUDENTS, model.CLASSES, domain=pyo.Binary)

    # --- Constraints ---
    # 1. Each student assigned to exactly one class
    @model.Constraint(model.STUDENTS)
    def assign_rule(m, s):
        return sum(m.x[s, c] for c in m.CLASSES) == 1

    # 2. Class size limits
    @model.Constraint(model.CLASSES)
    def min_size_rule(m, c):
        return sum(m.x[s, c] for s in m.STUDENTS) >= class_min

    @model.Constraint(model.CLASSES)
    def max_size_rule(m, c):
        return sum(m.x[s, c] for s in m.STUDENTS) <= class_max

    # 3. Must-Separate pairs
    @model.Constraint(model.MUST_SEPARATE_PAIRS, model.CLASSES)
    def separate_rule(m, s1, s2, c):
        # If s1 and s2 are defined for the constraint for class c, they cannot both be 1
        return m.x[s1, c] + m.x[s2, c] <= 1

    # --- Objective Function (Dummy - just need feasibility) ---
    model.dummy_obj = pyo.Objective(expr=0, sense=pyo.minimize)

    # --- Solve ---
    allocation = None # Default to None
    try:
        # tee=True shows solver output in the console
        solver = SolverFactory(solver_name)
        results = solver.solve(model, tee=True)

        end_pyomo_time = time.time()
        print(f"Pyomo Solver ({solver_name}) finished in {end_pyomo_time - start_pyomo_time:.2f} seconds.")
        print(f"Solver Status: {results.solver.status}, Termination Condition: {results.solver.termination_condition}")

        # --- Extract Solution ---
        if results.solver.termination_condition == TerminationCondition.optimal or \
           results.solver.termination_condition == TerminationCondition.feasible:

            print("Pyomo found a feasible solution.")
            temp_allocation = [[] for _ in range(num_classes)]
            for s in model.STUDENTS:
                for c in model.CLASSES:
                    # Check variable value (use tolerance for potential float issues)
                    if pyo.value(model.x[s, c]) > 0.5:
                        temp_allocation[c].append(s)
                        break # Move to next student

            # Verify extracted allocation sizes (redundant check, but good practice)
            sizes = [len(cls) for cls in temp_allocation]
            if all(class_min <= s <= class_max for s in sizes):
                 allocation = temp_allocation # Assign the valid allocation
                 print("Solution successfully extracted and verified.")
            else:
                 print(f"Error: Pyomo solution extracted, but size constraints violated? Sizes: {sizes}. Returning None.")
                 # This suggests a potential issue in model definition or solver tolerance

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            print("Pyomo determined the problem is Infeasible with the given constraints.")
        else:
            # Other conditions (unbounded, error, etc.)
            print(f"Pyomo solver terminated with status: {results.solver.termination_condition}")

    except Exception as e:
        end_pyomo_time = time.time()
        print(f"!!! Error during Pyomo model solving after {end_pyomo_time - start_pyomo_time:.2f} seconds !!!")
        print(f"Error: {e}")
        print(f"-> Please ensure the solver '{solver_name}' is installed and accessible in your PATH,")
        print(f"-> OR that you provided the correct path if installed elsewhere.")
        print(f"-> If using conda, try: conda install -c conda-forge {solver_name} (e.g., coincbc)")
        allocation = None # Ensure allocation is None on error

    return allocation


# --- Genetic Algorithm Helper Functions (Modified Evaluation - Keep as is) ---
# create_random_allocation, create_heuristic_allocation, check_hard_constraints,
# evaluate_objectives_and_constraints, normalize_objectives, calculate_scalar_fitness
# are reused, but check_hard_constraints will still be used within the GA loop
# as a safety check and for fitness calculation, even though Pyomo provides initial feasibility.

# (Keep these functions as they were in the previous hybrid version)
def create_random_allocation(df, num_classes):
    """Creates a valid random allocation (list of lists)."""
    student_ids = df['StudentID'].tolist()
    random.shuffle(student_ids)
    k, m = divmod(len(student_ids), num_classes)
    classes = [student_ids[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_classes)]
    return classes

def create_heuristic_allocation(df, num_classes):
    """Creates an allocation based on simple heuristics (balancing academic, language)."""
    num_students = len(df)
    df_sorted = df.sort_values(by=['Academic_Performance', 'language'], ascending=[False, True]).reset_index(drop=True)
    sorted_student_ids = df_sorted['StudentID'].tolist()
    allocation = [[] for _ in range(num_classes)]
    for i, student_id in enumerate(sorted_student_ids):
        class_id = i % num_classes
        allocation[class_id].append(student_id)
    return allocation

def check_hard_constraints(allocation, must_separate_pairs, class_min, class_max, student_data_map):
    """Checks if an allocation violates hard constraints."""
    student_to_class = {sid: i for i, cls in enumerate(allocation) for sid in cls}

    # Check class sizes
    for i, cls in enumerate(allocation):
        if not (class_min <= len(cls) <= class_max):
            return False # Size violation

    # Check must-separate pairs
    valid_student_set = set(student_data_map.keys()) # Get students actually in the data map
    for s1_id, s2_id in must_separate_pairs:
         # Check if both students are in the allocation and in the same class
         if s1_id in student_to_class and s2_id in student_to_class: # Check if they exist in current allocation
              if s1_id in valid_student_set and s2_id in valid_student_set: # Check they exist in data map
                 if student_to_class[s1_id] == student_to_class[s2_id]:
                      return False # Separation violation

    return True # All hard constraints satisfied

def evaluate_objectives_and_constraints(allocation, df, student_data_map, friends_map, must_separate_pairs, class_min, class_max):
    """
    Calculates objective values *if* hard constraints are met.
    Returns (is_feasible, objective_values_dict or None)
    """
    num_students = len(df)
    if not num_students: return False, None # Handle empty df case

    # 1. Check Hard Constraints first
    # student_data_map is needed for the check function now
    is_feasible = check_hard_constraints(allocation, must_separate_pairs, class_min, class_max, student_data_map)

    if not is_feasible:
        return False, None # Return immediately if infeasible

    # 2. Calculate Soft Objectives (only if feasible)
    # ... (Objective calculation logic remains the same as before) ...
    objective_values = {}
    num_classes = len(allocation)

    # --- Objective 1: Academic Equity ---
    class_academic_means = [np.mean([student_data_map[sid]['Academic_Performance'] for sid in cls]) for cls in allocation if cls]
    academic_variance = np.var(class_academic_means) if len(class_academic_means) > 1 else 0.0
    objective_values['academic_variance'] = academic_variance

    # --- Objective 2: Well-Being Balance ---
    class_wellbeing_risks = [np.mean([student_data_map[sid]['Wellbeing_Risk'] for sid in cls]) for cls in allocation if cls]
    wellbeing_risk_variance = np.var(class_wellbeing_risks) if len(class_wellbeing_risks) > 1 else 0.0
    objective_values['wellbeing_risk_variance'] = wellbeing_risk_variance

    # --- Objective 3: Social Cohesion ---
    intra_class_friendships = 0
    antagonistic_classes_count = 0
    class_assignments = {sid: i for i, cls in enumerate(allocation) for sid in cls}

    for class_id, cls in enumerate(allocation):
        class_bullies = [sid for sid in cls if student_data_map[sid]['Is_Bully'] == 1]
        class_vulnerables = [sid for sid in cls if student_data_map[sid]['Is_Vulnerable'] == 1]
        has_bully_vulnerable_conflict = len(class_bullies) > 0 and len(class_vulnerables) > 0
        has_multiple_bullies = len(class_bullies) > 1
        if has_bully_vulnerable_conflict: antagonistic_classes_count += 1
        if has_multiple_bullies: antagonistic_classes_count += 1

        for student_id in cls:
            friends_str = friends_map.get(student_id, "")
            if friends_str and isinstance(friends_str, str):
                friends_list = [f.strip() for f in friends_str.split(',') if f.strip()]
                for friend_id in friends_list:
                    if friend_id in class_assignments and class_assignments[friend_id] == class_id:
                        intra_class_friendships += 0.5

    social_cohesion_score = (intra_class_friendships - antagonistic_classes_count) / num_students if num_students > 0 else 0.0
    objective_values['social_cohesion'] = social_cohesion_score

    return True, objective_values


def normalize_objectives(objective_values_list):
    """Normalize objective values across the population to 0-1 (higher is better)."""
    # ... (No changes needed in this function) ...
    valid_objective_values = [ov for ov in objective_values_list if ov is not None]
    if not valid_objective_values:
        return [None] * len(objective_values_list)

    normalized_values_list = []
    min_max_values = {}
    objective_keys = valid_objective_values[0].keys()

    for key in objective_keys:
        values = [values_dict.get(key, 0) for values_dict in valid_objective_values]
        min_max_values[key] = (min(values), max(values))

    original_index = 0
    for i in range(len(objective_values_list)):
        values = objective_values_list[i]
        if values is None:
            normalized_values_list.append(None)
            continue

        normalized_values = {}
        # Need to access the *original* value dict corresponding to the current valid one
        current_valid_values = valid_objective_values[original_index]
        for key, value in current_valid_values.items(): # Use item from valid list for calc range
            min_val, max_val = min_max_values.get(key, (0, 1))
            current_value = values.get(key, 0) # Use actual value for normalization calculation
            if max_val == min_val:
                normalized_values[key] = 0.5
            else:
                if key in ['academic_variance', 'wellbeing_risk_variance']: # Minimize these
                    normalized_values[key] = 1.0 - ((current_value - min_val) / (max_val - min_val))
                elif key == 'social_cohesion': # Maximize this
                    normalized_values[key] = (current_value - min_val) / (max_val - min_val)
                else: # Default direct normalization
                    normalized_values[key] = (current_value - min_val) / (max_val - min_val)
        normalized_values_list.append(normalized_values)
        original_index += 1

    return normalized_values_list


def calculate_scalar_fitness(normalized_objective_values, weights, is_feasible):
    """Combines normalized objectives into a single fitness score. Applies penalty if not feasible."""
    # ... (No changes needed in this function) ...
    if not is_feasible or normalized_objective_values is None:
        return -CONSTRAINT_PENALTY
    fitness = 0.0
    for key, weight in weights.items():
         fitness += normalized_objective_values.get(key, 0) * weight
    return fitness + 1e-9


# --- GA Operators (Tournament, Crossover, Mutation - Keep as is) ---
def tournament_selection(population, fitness_scores, num_parents, tournament_size):
    # ... (No changes needed in this function) ...
    parents = []
    pop_indices = list(range(len(population)))
    current_tournament_size = min(tournament_size, len(pop_indices))
    if current_tournament_size < 1: return []
    for _ in range(num_parents):
        tournament_indices = random.sample(pop_indices, current_tournament_size)
        winner_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        parents.append(population[winner_index])
    return parents

def crossover(parent1_alloc, parent2_alloc, df):
    # ... (No changes needed in this function) ...
    num_classes = len(parent1_alloc)
    num_students = len(df)
    parent1_map = {sid: i for i, cls in enumerate(parent1_alloc) for sid in cls}
    parent2_map = {sid: i for i, cls in enumerate(parent2_alloc) for sid in cls}
    temp_assignment = [None] * num_students
    student_ids = df['StudentID'].tolist()

    for i, student_id in enumerate(student_ids):
        p1_class = parent1_map.get(student_id, random.randrange(num_classes))
        p2_class = parent2_map.get(student_id, random.randrange(num_classes))
        chosen_class = random.choice([p1_class, p2_class])
        temp_assignment[i] = chosen_class

    offspring_alloc = [[] for _ in range(num_classes)]
    current_sizes = {i: temp_assignment.count(i) for i in range(num_classes)}
    target_size_base = num_students // num_classes
    target_sizes = {i: target_size_base + (1 if i < (num_students % num_classes) else 0) for i in range(num_classes)}

    for i, student_id in enumerate(student_ids):
        offspring_alloc[temp_assignment[i]].append(student_id)

    overloaded_classes = sorted([i for i, size in current_sizes.items() if size > target_sizes[i]], key=lambda i: current_sizes[i], reverse=True)
    underloaded_classes = sorted([i for i, size in current_sizes.items() if size < target_sizes[i]], key=lambda i: current_sizes[i])

    while overloaded_classes and underloaded_classes:
        over_class_id = overloaded_classes[0]
        under_class_id = underloaded_classes[0]
        num_to_move = min(current_sizes[over_class_id] - target_sizes[over_class_id], target_sizes[under_class_id] - current_sizes[under_class_id])
        if num_to_move <= 0: break

        students_in_over_class = offspring_alloc[over_class_id]
        students_to_move_ids = random.sample(students_in_over_class, min(num_to_move, len(students_in_over_class)))

        for student_id_to_move in students_to_move_ids:
            if student_id_to_move in offspring_alloc[over_class_id]:
                 offspring_alloc[over_class_id].remove(student_id_to_move)
                 offspring_alloc[under_class_id].append(student_id_to_move)
                 current_sizes[over_class_id] -= 1
                 current_sizes[under_class_id] += 1

        if current_sizes[over_class_id] <= target_sizes[over_class_id]:
            overloaded_classes.pop(0)
        if current_sizes[under_class_id] >= target_sizes[under_class_id]:
             underloaded_classes.pop(0)

    return offspring_alloc

def mutate(allocation, mutation_rate):
    # ... (No changes needed in this function) ...
    num_classes = len(allocation)
    num_students_total = sum(len(cls) for cls in allocation)
    mutated_alloc = [list(cls) for cls in allocation]
    num_swap_operations = int(num_students_total * mutation_rate / 2)

    for _ in range(num_swap_operations):
        valid_classes = [i for i, cls in enumerate(mutated_alloc) if len(cls) > 0]
        if len(valid_classes) < 2: break
        class1_id, class2_id = random.sample(valid_classes, 2)

        if mutated_alloc[class1_id] and mutated_alloc[class2_id]:
            student1_id = random.choice(mutated_alloc[class1_id])
            student2_id = random.choice(mutated_alloc[class2_id])

            mutated_alloc[class1_id].remove(student1_id)
            mutated_alloc[class1_id].append(student2_id)
            mutated_alloc[class2_id].remove(student2_id)
            mutated_alloc[class2_id].append(student1_id)

    return mutated_alloc

# --- Main Hybrid Allocation Function ---

def run_hybrid_allocation(df):
    """Runs the Hybrid Pyomo-seeded GA to find an optimal classroom allocation."""
    # --- Use Pyomo for Seeding ---
    print(f"Starting Hybrid Allocation with {NUM_STUDENTS} students, {N_CLASSES} classes...")
    print(f"Target Class Size: {CLASS_SIZE_TARGET}, Allowed Range: [{CLASS_SIZE_MIN}, {CLASS_SIZE_MAX}]")
    start_time = time.time()

    num_students = len(df)
    num_classes = N_CLASSES
    num_elitism = max(1, int(GA_POP_SIZE * GA_ELITISM_RATE))

    # Pre-process data
    student_data_map = df.set_index('StudentID').to_dict(orient='index')
    friends_map = df.set_index('StudentID')['Friends'].to_dict()

    # Identify must-separate pairs
    bullies = df[df['Is_Bully'] == 1]['StudentID'].tolist()
    vulnerables = df[df['Is_Vulnerable'] == 1]['StudentID'].tolist()
    must_separate_pairs = []
    for b in bullies:
        for v in vulnerables:
            if b != v:
                must_separate_pairs.append(tuple(sorted((b, v)))) # Store consistently sorted tuple
    must_separate_pairs = list(set(must_separate_pairs)) # Keep unique pairs
    print(f"Identified {len(must_separate_pairs)} unique bully-vulnerable pairs to separate.")

    # 1. Population Initialization (Pyomo Seeding + Random/Heuristic)
    population = []

    # Try to generate feasible solutions using Pyomo
    for i in range(PYOMO_SEED_COUNT):
        print(f"\n--- Pyomo Seed Attempt {i+1}/{PYOMO_SEED_COUNT} ---")
        # Pass the chosen solver name from config
        pyomo_solution = generate_pyomo_feasible_solution(
            df, num_classes, must_separate_pairs, CLASS_SIZE_MIN, CLASS_SIZE_MAX, solver_name=PYOMO_SOLVER
        )
        if pyomo_solution:
            population.append(pyomo_solution)
            print(f"Pyomo Seed {i+1} added to population.")
        else:
            print(f"Pyomo Seed {i+1} failed.")
            if i == 0 and PYOMO_SEED_COUNT > 0:
                 print("\nWARNING: Pyomo could not find even one initial feasible solution.")
                 print("This might indicate conflicting constraints or solver issues.")
                 print("The GA will proceed with random/heuristic initialization, but may struggle.")

    num_pyomo_seeds = len(population)
    print(f"\n--- Seeding GA Population ---")
    print(f"Successfully seeded {num_pyomo_seeds} individuals using Pyomo.")

    # Fill the rest of the population
    num_to_fill = GA_POP_SIZE - num_pyomo_seeds
    num_heuristic = min(num_to_fill // 2, num_to_fill)
    num_random = num_to_fill - num_heuristic

    print(f"Adding {num_heuristic} heuristic individuals...")
    for _ in range(num_heuristic):
        population.append(create_heuristic_allocation(df, num_classes))

    print(f"Adding {num_random} random individuals...")
    while len(population) < GA_POP_SIZE:
        population.append(create_random_allocation(df, num_classes))

    print(f"Initialized population of {len(population)}.")

    # --- GA Loop (remains largely the same logic as before) ---
    best_allocation_overall = None
    best_fitness_overall = -float('inf')
    best_objectives_overall = None
    fitness_history = []
    best_fitness_history_for_improvement_check = []

    for generation in range(GA_NUM_GENERATIONS):
        gen_start_time = time.time()
        # 2. Fitness Evaluation (Checks constraints internally)
        eval_results = [evaluate_objectives_and_constraints(alloc, df, student_data_map, friends_map, must_separate_pairs, CLASS_SIZE_MIN, CLASS_SIZE_MAX) for alloc in population]
        feasibility_flags = [res[0] for res in eval_results]
        objective_values_pop = [res[1] for res in eval_results]

        # Normalize objectives
        normalized_objective_values_pop = normalize_objectives(objective_values_pop)

        # Calculate scalar fitness
        fitness_scores = [calculate_scalar_fitness(norm_obj, FITNESS_WEIGHTS, is_feasible)
                          for norm_obj, is_feasible in zip(normalized_objective_values_pop, feasibility_flags)]

        num_feasible = sum(feasibility_flags)
        avg_fitness = np.mean([f for f, feasible in zip(fitness_scores, feasibility_flags) if feasible]) if num_feasible > 0 else -CONSTRAINT_PENALTY
        max_fitness = np.max(fitness_scores) if fitness_scores else -CONSTRAINT_PENALTY

        # Identify current best *feasible* individual
        current_best_fitness_gen = -float('inf')
        current_best_alloc_gen = None
        current_best_objectives_gen = None
        feasible_indices = [i for i, feasible in enumerate(feasibility_flags) if feasible]

        if feasible_indices:
             current_best_idx_gen = max(feasible_indices, key=lambda i: fitness_scores[i])
             current_best_fitness_gen = fitness_scores[current_best_idx_gen]
             current_best_alloc_gen = population[current_best_idx_gen]
             current_best_objectives_gen = objective_values_pop[current_best_idx_gen]

        # Update overall best
        if current_best_alloc_gen is not None and current_best_fitness_gen > best_fitness_overall:
            best_fitness_overall = current_best_fitness_gen
            best_allocation_overall = current_best_alloc_gen
            best_objectives_overall = current_best_objectives_gen
            print(f"Generation {generation}: New best FEASIBLE fitness = {best_fitness_overall:.4f} (Avg Fitness: {avg_fitness:.4f}, Feasible: {num_feasible}/{len(population)})")
            if best_objectives_overall:
                obj_str = ", ".join([f"{k}: {v:.4f}" for k, v in best_objectives_overall.items()])
                print(f"  Objectives: {obj_str}")
        elif generation % 10 == 0 or generation == GA_NUM_GENERATIONS - 1:
             print(f"Generation {generation}: Max Fitness = {max_fitness:.4f} (Avg Fitness: {avg_fitness:.4f}, Feasible: {num_feasible}/{len(population)})")

        fitness_history.append(max_fitness)
        best_fitness_history_for_improvement_check.append(best_fitness_overall if best_fitness_overall > -float('inf') else max_fitness)

        # --- Adaptive Mutation ---
        fitness_variance = np.var(fitness_scores) if len(fitness_scores) > 1 else 0.0
        current_mutation_rate = GA_MUTATION_RATE_HIGH if fitness_variance < FITNESS_VARIANCE_THRESHOLD else GA_MUTATION_RATE_LOW

        # --- Termination Check ---
        if generation >= IMPROVEMENT_CHECK_GENERATIONS:
            recent_best_feasible = [f for f in best_fitness_history_for_improvement_check[-IMPROVEMENT_CHECK_GENERATIONS:] if f > -CONSTRAINT_PENALTY]
            if recent_best_feasible and best_fitness_overall > -CONSTRAINT_PENALTY:
                 improvement = best_fitness_overall - min(recent_best_feasible)
                 if improvement < FITNESS_IMPROVEMENT_THRESHOLD:
                    print(f"Termination: Best feasible fitness improvement ({improvement:.6f}) below threshold.")
                    break
            elif len(recent_best_feasible) == 0 and generation > IMPROVEMENT_CHECK_GENERATIONS * 2:
                 print(f"Termination: No feasible solution found for {IMPROVEMENT_CHECK_GENERATIONS} generations. Stopping.")
                 break

        # --- Selection and Reproduction ---
        top_indices = np.argsort(fitness_scores)[-num_elitism:][::-1]
        next_population = [population[i] for i in top_indices]
        parents = tournament_selection(population, fitness_scores, GA_POP_SIZE - num_elitism, GA_TOURNAMENT_SIZE)

        offspring = []
        num_offspring_needed = GA_POP_SIZE - len(next_population)
        current_parents_pool = parents

        if not current_parents_pool and num_offspring_needed > 0:
             print(f"Warning: No parents selected in generation {generation}. Adding random individuals.")
             while len(next_population) < GA_POP_SIZE:
                  next_population.append(create_random_allocation(df, num_classes))
        else:
            while len(offspring) < num_offspring_needed:
                 if len(current_parents_pool) >= 2: p1, p2 = random.sample(current_parents_pool, 2)
                 elif len(current_parents_pool) == 1: p1 = p2 = current_parents_pool[0]
                 else: break # Should not happen
                 child = crossover(p1, p2, df)
                 mutated_child = mutate(child, current_mutation_rate)
                 offspring.append(mutated_child)

        next_population.extend(offspring)
        population = next_population[:GA_POP_SIZE] # Ensure population size doesn't exceed limit

        gen_end_time = time.time()
        # print(f"Generation {generation} took {gen_end_time - gen_start_time:.2f} seconds.")

    # --- Final Result Processing (remains the same) ---
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nHybrid Allocation finished after {generation + 1} generations in {duration:.2f} seconds.")

    if best_allocation_overall is None:
         print("\n>>> CRITICAL WARNING: No feasible solution satisfying all constraints was found by the hybrid algorithm. <<<")
         final_best_idx = np.argmax(fitness_scores) if fitness_scores else 0
         if final_best_idx < len(population):
            final_best_alloc = population[final_best_idx]
            final_best_fitness = fitness_scores[final_best_idx] if fitness_scores else -float('inf')
            print(f"   Returning the 'best' found allocation (Fitness: {final_best_fitness:.4f}), but it likely violates constraints.")
            # Perform constraint check on this allocation to report violations
            is_final_feasible = check_hard_constraints(final_best_alloc, must_separate_pairs, CLASS_SIZE_MIN, CLASS_SIZE_MAX, student_data_map)
            if not is_final_feasible: print("   Constraint check confirms final allocation is INFEASIBLE.")
            best_allocation_overall = final_best_alloc
         else:
            print("   Error: Could not retrieve any allocation to return.")
            best_allocation_overall = None # No allocation found
         best_objectives_overall = {'academic_variance': float('nan'), 'wellbeing_risk_variance': float('nan'), 'social_cohesion': float('nan')}

    else:
        print(f"Best feasible fitness found: {best_fitness_overall:.4f}")
        print("Best Feasible Allocation Objectives:")
        if best_objectives_overall:
            for k, v in best_objectives_overall.items():
                print(f"  {k}: {v:.4f}")
        else:
            print("  (No objectives calculated - indicates an issue)")

    # Convert best allocation to DataFrame
    allocated_student_list = []
    if best_allocation_overall:
        for class_id, student_list in enumerate(best_allocation_overall):
            for student_id in student_list:
                allocated_student_list.append({'StudentID': student_id, 'Allocated_Class': f"Class_{class_id+1}"})
    else: # Fallback if absolutely no allocation is available
        student_ids = df['StudentID'].tolist()
        for student_id in student_ids:
             allocated_student_list.append({'StudentID': student_id, 'Allocated_Class': f"Unassigned_ERROR"})

    allocation_df = pd.DataFrame(allocated_student_list)
    final_df = pd.merge(df, allocation_df, on='StudentID', how='left')
    final_df['Allocated_Class'] = final_df['Allocated_Class'].fillna('Unassigned')

    # --- Analysis/Visualization (remains the same) ---
    plot_html = "<p>Analysis requires a feasible solution.</p>"
    stats_df = pd.DataFrame()
    if best_allocation_overall and best_fitness_overall > -CONSTRAINT_PENALTY:
         # Recalculate final stats only if best solution was deemed feasible
         final_feasible, final_objectives = evaluate_objectives_and_constraints(best_allocation_overall, df, student_data_map, friends_map, must_separate_pairs, CLASS_SIZE_MIN, CLASS_SIZE_MAX)

         if final_feasible and final_objectives:
             print("\n--- Analysis of Best Feasible Allocation ---")
             class_stats = []
             for i, cls_list in enumerate(best_allocation_overall):
                 class_name = f"Class_{i+1}"
                 class_df = final_df[final_df['Allocated_Class'] == class_name]
                 if not class_df.empty: # Ensure class DF is not empty before calculating stats
                    stats = {
                        'Class': class_name, 'Size': len(cls_list),
                        'Avg_Academic': class_df['Academic_Performance'].mean(),
                        'Avg_Wellbeing_Risk': class_df['Wellbeing_Risk'].mean(),
                        'Num_Bullies': class_df['Is_Bully'].sum(),
                        'Num_Vulnerable': class_df['Is_Vulnerable'].sum(),
                        'Num_Supportive': class_df['Is_Supportive'].sum(),
                    }
                    class_stats.append(stats)
                 else: # Handle empty class case if it somehow occurs
                     class_stats.append({'Class': class_name, 'Size': 0})

             stats_df = pd.DataFrame(class_stats).fillna(0) # FillNa just in case means were calculated on empty dfs
             print(stats_df.round(2))

             print("\nObjective Values (Recalculated):")
             print(f"  Academic Variance (Minimize): {final_objectives.get('academic_variance', float('nan')):.4f}")
             print(f"  Wellbeing Risk Variance (Minimize): {final_objectives.get('wellbeing_risk_variance', float('nan')):.4f}")
             print(f"  Social Cohesion (Maximize): {final_objectives.get('social_cohesion', float('nan')):.4f}")

             # Generate plot
             try:
                plt.figure(figsize=(10, 6))
                stats_df.plot(kind='bar', x='Class', y=['Avg_Academic', 'Avg_Wellbeing_Risk'], secondary_y=['Avg_Wellbeing_Risk'], rot=0)
                plt.title('Class Profiles (Best Allocation)')
                plt.ylabel('Average Score / Risk')
                plt.xlabel('Class')
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                plot_html = f"<img src='data:image/png;base64,{plot_base64}'/>"
             except Exception as e:
                 print(f"Error generating plot: {e}")
                 plot_html = "<p>Error generating plot.</p>"
         else:
              print("\nCould not recalculate statistics for the final allocation.")
              plot_html = "<p>Could not generate analysis for the final allocation.</p>"

    else: # Handle case where no feasible solution was ever found
         print("\n--- No Feasible Allocation Found ---")
         stats_df = pd.DataFrame([{"Status": "No Feasible Solution Found"}])
         plot_html = "<p>No feasible allocation found.</p>"


    return final_df, stats_df, plot_html


# --- Gradio Interface (remains the same) ---
def run_allocation_interface(run_button_click):
    if not run_button_click:
        return pd.DataFrame(), pd.DataFrame(), "<p>Click 'Run Allocation' to start.</p>", "<p></p>"

    print("\n--- Running Full Process ---")
    student_df = generate_synthetic_data(num_students=NUM_STUDENTS)
    if student_df is None or student_df.empty:
        return pd.DataFrame(), pd.DataFrame(), "<p>Error generating/loading data.</p>", "<p></p>"

    student_df = run_predictive_analysis(student_df)
    if 'Wellbeing_Risk' not in student_df.columns:
         return student_df, pd.DataFrame(), "<p>Error during predictive analysis.</p>", "<p></p>"

    try:
        final_df, stats_df, plot_html = run_hybrid_allocation(student_df)
        stats_html = stats_df.to_html(index=False, escape=False, classes='table table-striped table-sm')
        return final_df, stats_df, plot_html, stats_html
    except Exception as e:
        print(f"An error occurred during the hybrid allocation: {e}")
        import traceback
        traceback.print_exc()
        return student_df, pd.DataFrame(), f"<p>Error during allocation: {e}</p>", "<p></p>"

# Define Gradio components
run_button = gr.Button("Run Allocation", variant="primary")
output_df = gr.DataFrame(label="Final Allocation with Student Data")
output_stats_df = gr.DataFrame(label="Class Statistics Summary (Raw Data)") # Hidden
output_plot = gr.HTML(label="Class Profiles Visualization")
output_stats_html = gr.HTML(label="Class Statistics Summary")

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Hybrid Pyomo-GA Student Class Allocation Tool
        This tool uses a hybrid approach combining **Pyomo (with a backend solver like CBC/GLPK)** and a **Genetic Algorithm (GA)**
        to allocate students into classes.
        - **Pyomo:** Enforces hard constraints (class size, separating specific students) by finding initial feasible solutions.
        - **GA:** Optimizes soft objectives (academic equity, wellbeing balance, social cohesion).
        Click 'Run Allocation' to start. **Requires Pyomo and a compatible solver (e.g., CBC) installed.**
        """
    )
    run_button.click(
        fn=run_allocation_interface,
        inputs=[run_button],
        outputs=[output_df, output_stats_df, output_plot, output_stats_html]
    )
    with gr.Row():
        output_stats_html
    with gr.Row():
        output_plot
    with gr.Accordion("Show Full Data with Allocations", open=False):
         output_df

# Launch the Gradio app
if __name__ == "__main__":
    print(f"\n--- Configuration Summary ---")
    print(f"Num Students: {NUM_STUDENTS}, Target Class Size: {CLASS_SIZE_TARGET}, Num Classes: {N_CLASSES}")
    print(f"Class Size Constraints: Min={CLASS_SIZE_MIN}, Max={CLASS_SIZE_MAX}")
    print(f"GA Pop Size: {GA_POP_SIZE}, Generations: {GA_NUM_GENERATIONS}")
    print(f"Pyomo Seeds Attempted: {PYOMO_SEED_COUNT}, Pyomo Solver: {PYOMO_SOLVER}") # Updated print
    print(f"Bully Threshold: {BULLY_CRITICISES_THRESHOLD}, Vulnerable Quantile: {VULNERABLE_WELLBEING_QUANTILE}")
    print(f"Fitness Weights: {FITNESS_WEIGHTS}")
    print(f"Constraint Penalty: {CONSTRAINT_PENALTY}")
    print("\n---> REMINDER: Ensure Pyomo is installed (`pip install pyomo`) <---")
    print(f"---> AND ensure the solver '{PYOMO_SOLVER}' is installed and accessible (e.g., via conda) <---")
    print("\nLaunching Gradio Interface...")
    demo.launch()