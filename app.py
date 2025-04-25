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

# --- Configuration & Setup ---
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
plt.style.use('seaborn-v0_8-whitegrid')

SYNTHETIC_DATA_CSV = "synthetic_student_data.csv"
NUM_STUDENTS = 10000 # Consider increasing for larger scale testing (e.g., 5000, 10000)
CLASS_SIZE_TARGET = 25
N_CLASSES = max(1, round(NUM_STUDENTS / CLASS_SIZE_TARGET)) # Renamed to N_CLASSES for clarity

# Define thresholds and parameters for Objectives/GA
BULLY_CRITICISES_THRESHOLD = 6 # Students with criticises score >= this are potential bullies
VULNERABLE_WELLBEING_QUANTILE = 0.8 # Students in top 20% for Wellbeing_Risk are vulnerable

# Genetic Algorithm ParametersS
GA_POP_SIZE = 500 # Population size (suggested 500-1000)
GA_NUM_GENERATIONS = 100 # Number of generations (suggested 100)
GA_ELITISM_RATE = 0.05 # Percentage of top individuals to preserve (suggested 5%)
GA_MUTATION_RATE_LOW = 0.02 # Base mutation rate (suggested 2%)
GA_MUTATION_RATE_HIGH = 0.05 # Increased mutation rate (suggested 5%)
GA_TOURNAMENT_SIZE = 5 # For parent selection
GA_HEURISTIC_SEED_PERCENT = 0.20 # Percentage of population seeded heuristically (suggested 20%)

# Adaptive Mutation Threshold
# If fitness variance drops below this, increase mutation rate
FITNESS_VARIANCE_THRESHOLD = 0.0005 # Example threshold, may need tuning

# Termination Criteria Thresholds
FITNESS_IMPROVEMENT_THRESHOLD = 0.001 # Stop if best fitness improves by less than this
IMPROVEMENT_CHECK_GENERATIONS = 10 # Check improvement over this many generations


# Weights for the *scalar* fitness function (how we combine objectives for the GA)
# These weights determine which point on the approximate Pareto front the GA converges towards.
# Objectives will be normalized 0-1, then combined.
FITNESS_WEIGHTS = {
    'academic_equity': 2.0, # Maximize 1/variance
    'wellbeing_balance': 1.5, # Maximize 1/variance (using variance of average wellbeing risk)
    'social_cohesion': 3.0, # Maximize (Friends - Conflicts)/N
}


# --- Data Generation (Keep as is, generates necessary raw features) ---
def generate_synthetic_data(filename=SYNTHETIC_DATA_CSV, num_students=NUM_STUDENTS):
    """Generates synthetic student data if the CSV doesn't exist or is invalid."""
    if os.path.exists(filename):
        print(f"Loading existing data from {filename}")
        try:
            df = pd.read_csv(filename)
            # Basic checks: correct number of students and essential columns
            required_cols = ['StudentID', 'Academic_Performance', 'Friends', 'criticises', 'k6_overall', 'School_support_engage', 'language']
            if len(df) == num_students and all(col in df.columns for col in required_cols):
                print("Data loaded successfully.")
                return df
            else:
                print("CSV found but invalid (row count mismatch or missing columns), regenerating...")
        except Exception as e:
            print(f"Error loading CSV: {e}. Regenerating...")

    print(f"Generating {num_students} new synthetic student records...")
    # Define possible scales based on survey info
    LIKERT_SCALE_1_7 = list(range(1, 8))
    K6_SCALE_1_5 = list(range(1, 6))
    LANGUAGE_SCALE = [0, 1] # 0: Primary language (e.g., English), 1: Other language
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
            "language": random.choices(LANGUAGE_SCALE, weights=[0.8, 0.2], k=1)[0], # 80% primary language, 20% other
            "COVID": random.choice(LIKERT_SCALE_1_7),
            "criticises": random.choice(LIKERT_SCALE_1_7), # Key for bullying
            "MenBetterSTEM": random.choice(LIKERT_SCALE_1_7),
            "pwi_wellbeing": random.choice(PWI_SCALE), # Key for wellbeing
            "Intelligence1": random.choice(LIKERT_SCALE_1_7),
            "Intelligence2": random.choice(LIKERT_SCALE_1_7),
            "Soft": random.choice(LIKERT_SCALE_1_7),
            "opinion": random.choice(LIKERT_SCALE_1_7),
            "Nerds": random.choice(LIKERT_SCALE_1_7),
            "comfortable": random.choice(LIKERT_SCALE_1_7),
            "future": random.choice(LIKERT_SCALE_1_7),
            "bullying": random.choice(LIKERT_SCALE_1_7), # Another bullying indicator
             **{f"Manbox5_{i}": random.choice(LIKERT_SCALE_1_7) for i in range(1, 6)},
             **{f"k6_{i}": random.choice(K6_SCALE_1_5) for i in range(1, 7)}, # Key for wellbeing (K6)
        }
        data.append(student_data)

    df = pd.DataFrame(data)

    # --- Calculate Derived Fields ---
    df['Manbox5_overall'] = df[[f"Manbox5_{i}" for i in range(1, 6)]].mean(axis=1)
    df['Masculinity_contrained'] = df[['Soft', 'WomenDifferent', 'Nerds', 'MenBetterSTEM']].mean(axis=1)
    df['GrowthMindset'] = ((8.0 - df['Intelligence1']) + (8.0 - df['Intelligence2'])) / 2.0
    df['k6_overall'] = df[[f"k6_{i}" for i in range(1, 7)]].sum(axis=1) # Higher is worse wellbeing
    df['School_support_engage'] = (df['comfortable'] + df['future'] + (8.0 - df['isolated']) + (8.0 - df['opinion'])) / 4.0 # Higher is better

    # Generate Friends string last, ensuring all students exist
    df['Friends'] = df['StudentID'].apply(
        lambda x: ", ".join(random.sample([pid for pid in student_ids if pid != x], k=random.randint(0, min(7, num_students - 1)))) # Ensure k is valid
    )
    df['Friends_Count'] = df['Friends'].fillna('').apply(lambda x: len([f for f in x.split(',') if f.strip()]))

    df.to_csv(filename, index=False)
    print(f"Synthetic data saved to {filename}")
    return df

# --- Predictive Analysis (Kept as it provides useful input features/scores for the GA objectives) ---
def run_predictive_analysis(df):
    """Runs predictive models to get risk/score indicators and flags."""
    print("Running predictive analytics for risk scores...")

    # Ensure all required raw columns exist, fill NaNs if necessary for calculations
    raw_cols_needed = [
        'Academic_Performance', 'isolated', 'WomenDifferent', 'language',
        'pwi_wellbeing', 'Intelligence1', 'Intelligence2', 'Soft', 'opinion',
        'Nerds', 'MenBetterSTEM', 'comfortable', 'future', 'bullying', 'criticises',
    ] + [f"Manbox5_{i}" for i in range(1, 6)] + [f"k6_{i}" for i in range(1, 7)]

    for col in raw_cols_needed:
        if col not in df.columns:
            print(f"Warning: Raw feature '{col}' not found. Adding with 0.")
            df[col] = 0
        elif df[col].isnull().any():
            # Attempt conversion before filling to avoid errors if mixed types
            try:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 df[col] = df[col].fillna(df[col].median())
            except Exception as e:
                 print(f"Could not convert column {col} to numeric: {e}. Filling with 0.")
                 df[col] = df[col].fillna(0)

    # Recalculate derived fields using cleaned raw data
    df['Manbox5_overall'] = df[[f"Manbox5_{i}" for i in range(1, 6)]].mean(axis=1)
    df['Masculinity_contrained'] = df[['Soft', 'WomenDifferent', 'Nerds', 'MenBetterSTEM']].mean(axis=1)
    df['GrowthMindset'] = ((8.0 - df['Intelligence1']) + (8.0 - df['Intelligence2'])) / 2.0
    df['k6_overall'] = df[[f"k6_{i}" for i in range(1, 7)]].sum(axis=1)
    df['School_support_engage'] = (df['comfortable'] + df['future'] + (8.0 - df['isolated']) + (8.0 - df['opinion'])) / 4.0
    df['Friends_Count'] = df['Friends'].fillna('').apply(lambda x: len([f for f in x.split(',') if f.strip()]))


    # Simulate Labels (for training predictive models)
    df['Academic_Success'] = (df['Academic_Performance'] > df['Academic_Performance'].quantile(0.75)).astype(int)
    df['Wellbeing_Decline'] = (df['k6_overall'] > df['k6_overall'].quantile(0.75)).astype(int) # Higher k6 is worse
    df['Positive_Peer_Collab'] = (df['Friends_Count'] > df['Friends_Count'].median()).astype(int)

    # Features for prediction
    features_for_prediction = [
        'Academic_Performance', 'isolated', 'WomenDifferent', 'language',
        'pwi_wellbeing', 'GrowthMindset', 'k6_overall', 'Manbox5_overall',
        'Masculinity_contrained', 'School_support_engage', 'Friends_Count'
    ]

    # Ensure features exist and handle NaNs after initial processing
    X = df[features_for_prediction].copy()
    for col in X.columns:
        if X[col].isnull().any():
            # print(f"Warning: Feature '{col}' has NaNs before prediction. Filling with median.")
            X[col] = X[col].fillna(X[col].median())
        if np.isinf(X[col]).any():
             # print(f"Warning: Feature '{col}' has Infs before prediction. Replacing with NaN then filling median.")
             X[col] = X[col].replace([np.inf, -np.inf], np.nan)
             X[col] = X[col].fillna(X[col].median())

    y_academic = df['Academic_Success']
    y_wellbeing = df['Wellbeing_Decline']
    y_peer = df['Positive_Peer_Collab']

    # Handle cases where a target class might have only one label
    try:
        if len(y_academic.unique()) > 1:
            academic_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(X, y_academic)
            df['Academic_Risk'] = 1 - academic_model.predict_proba(X)[:, 1] # P(not succeeding) = 1 - P(succeeding)
        else:
            df['Academic_Risk'] = 1 - y_academic.iloc[0] # Assign 0 or 1 based on the single label
            print("Warning: Academic_Success has only one unique value, skipping XGBoost training.")

        if len(y_wellbeing.unique()) > 1:
            wellbeing_model = RandomForestClassifier(random_state=42).fit(X, y_wellbeing)
            df['Wellbeing_Risk'] = wellbeing_model.predict_proba(X)[:, 1] # P(decline)
        else:
             df['Wellbeing_Risk'] = y_wellbeing.iloc[0] # Assign 0 or 1 based on the single label
             print("Warning: Wellbeing_Decline has only one unique value, skipping RandomForest training.")

        if len(y_peer.unique()) > 1:
            peer_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(X, y_peer)
            df['Peer_Score'] = peer_model.predict_proba(X)[:, 1] # P(positive collaboration)
        else:
            df['Peer_Score'] = y_peer.iloc[0] # Assign 0 or 1 based on the single label
            print("Warning: Positive_Peer_Collab has only one unique value, skipping XGBoost training.")

    except Exception as e:
        print(f"Error during model training/prediction: {e}")
        df['Academic_Risk'] = 0.5
        df['Wellbeing_Risk'] = 0.5
        df['Peer_Score'] = 0.5

    # Identify Bully/Vulnerable Students (using criteria from previous version)
    df['Is_Bully'] = (df['criticises'] >= BULLY_CRITICISES_THRESHOLD).astype(int)
    # Handle case where Wellbeing_Risk might not have enough variance for quantile
    if df['Wellbeing_Risk'].nunique() > 1:
        wellbeing_risk_threshold = df['Wellbeing_Risk'].quantile(VULNERABLE_WELLBEING_QUANTILE)
    else:
         wellbeing_risk_threshold = df['Wellbeing_Risk'].iloc[0] # Use the single value if no variance

    df['Is_Vulnerable'] = (df['Wellbeing_Risk'] >= wellbeing_risk_threshold).astype(int)
    df['Is_Supportive'] = (df['School_support_engage'] >= df['School_support_engage'].quantile(0.8) if df['School_support_engage'].nunique() > 1 else df['School_support_engage'].iloc[0] >= df['School_support_engage'].iloc[0]).astype(int)


    print("Predictive analysis complete.")
    return df # Returns DataFrame with added risk scores and flags


# --- Genetic Algorithm Helper Functions ---

def create_random_allocation(df, num_classes):
    """Creates a valid random allocation (list of lists)."""
    student_ids = df['StudentID'].tolist()
    random.shuffle(student_ids)
    # Distribute students as evenly as possible
    k, m = divmod(len(student_ids), num_classes)
    classes = [student_ids[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_classes)]
    return classes

def create_heuristic_allocation(df, num_classes):
    """Creates an allocation based on simple heuristics (balancing academic, language)."""
    num_students = len(df)
    # Sort students based on a combination of features
    # Sort by Academic Performance descending, then language ascending
    df_sorted = df.sort_values(by=['Academic_Performance', 'language'], ascending=[False, True]).reset_index(drop=True)
    sorted_student_ids = df_sorted['StudentID'].tolist()

    allocation = [[] for _ in range(num_classes)]
    # Assign students round-robin from the sorted list to balance features
    for i, student_id in enumerate(sorted_student_ids):
        class_id = i % num_classes
        allocation[class_id].append(student_id)

    # Ensure class sizes are roughly equal after assignment (simple check)
    # This round-robin approach on a sorted list naturally creates balanced classes.
    return allocation


def evaluate_objectives(allocation, df, student_data_map, friends_map):
    """Calculates the values for the three core objectives."""
    num_classes = len(allocation)
    num_students = len(df)
    objective_values = {} # Use objective values, not scores yet

    # --- Objective 1: Academic Equity (Minimize Variance of Avg Academic Performance) ---
    class_academic_means = [np.mean([student_data_map[sid]['Academic_Performance'] for sid in cls]) for cls in allocation if cls]
    # Handle case with only one class or empty classes
    if len(class_academic_means) <= 1:
        academic_variance = 0.0
    else:
        academic_variance = np.var(class_academic_means)

    # We want to maximize 1/variance for academic equity. Minimize variance is equivalent for ranking.
    # Store variance directly, normalization will handle inversion for fitness.
    objective_values['academic_variance'] = academic_variance


    # --- Objective 2: Well-Being Balance (Minimize Variance of Avg Wellbeing Risk) ---
    # Using variance of average Wellbeing Risk across classes as a practical interpretation
    class_wellbeing_risks = [np.mean([student_data_map[sid]['Wellbeing_Risk'] for sid in cls]) for cls in allocation if cls]
    if len(class_wellbeing_risks) <= 1:
         wellbeing_risk_variance = 0.0
    else:
         wellbeing_risk_variance = np.var(class_wellbeing_risks)

    # Minimize variance of average Wellbeing Risk
    objective_values['wellbeing_risk_variance'] = wellbeing_risk_variance

    # Note: The Gini coefficient of within-class distributions is a different objective
    # and harder to integrate directly into this scalar fitness GA. Minimizing variance
    # of class averages across classes is a common proxy for balancing.


    # --- Objective 3: Social Cohesion ((Retained Friendships - Antagonistic Pairs) / Total Students) ---
    intra_class_friendships = 0
    antagonistic_classes_count = 0 # Count classes with conflicts

    class_assignments = {} # Map student ID to class ID for quick lookup
    for i, cls in enumerate(allocation):
        for sid in cls:
            class_assignments[sid] = i

    for class_id, cls in enumerate(allocation):
        class_bullies = [sid for sid in cls if student_data_map[sid]['Is_Bully'] == 1]
        class_vulnerables = [sid for sid in cls if student_data_map[sid]['Is_Vulnerable'] == 1]

        # Antagonistic Pairs Count (sum over classes of (1 if bully+vulnerable) + (1 if multiple bullies))
        has_bully_vulnerable_conflict = len(class_bullies) > 0 and len(class_vulnerables) > 0
        has_multiple_bullies = len(class_bullies) > 1
        if has_bully_vulnerable_conflict:
             antagonistic_classes_count += 1
        if has_multiple_bullies:
             antagonistic_classes_count += 1 # Count this as another type of conflict instance

        # Count Intra-Class Friendships in this class
        for student_id in cls:
            friends_str = friends_map.get(student_id, "")
            if friends_str and isinstance(friends_str, str): # Ensure it's a string before splitting
                friends_list = [f.strip() for f in friends_str.split(',') if f.strip()]
                for friend_id in friends_list:
                    # Check if friend is in the same class
                    if friend_id in class_assignments and class_assignments[friend_id] == class_id:
                        intra_class_friendships += 0.5 # Count each friendship once (A->B and B->A)

    # Calculate Social Cohesion Score as (Friends - Conflicts) / Total Students
    # Ensure total students is not zero if data generation failed
    social_cohesion_score = (intra_class_friendships - antagonistic_classes_count) / num_students if num_students > 0 else 0.0
    objective_values['social_cohesion'] = social_cohesion_score

    return objective_values

def normalize_objectives(objective_values_list):
    """Normalize objective values across the population to 0-1."""
    normalized_values_list = []
    # Find min/max for each objective across the population
    min_max_values = {}
    # Initialize with dummy values to handle cases where list might be empty or all values are the same
    if not objective_values_list:
        return []

    # Get keys from the first set of objective values
    objective_keys = objective_values_list[0].keys()

    for key in objective_keys:
        values = [values_dict.get(key, 0) for values_dict in objective_values_list] # Use .get with default 0
        min_max_values[key] = (min(values), max(values))

    # Normalize each value
    for values in objective_values_list:
        normalized_values = {}
        for key, value in values.items():
            min_val, max_val = min_max_values.get(key, (0, 1)) # Use .get with default min/max
            if max_val == min_val:
                normalized_values[key] = 0.5 # Handle zero range
            else:
                # Academic and Wellbeing Variance are MINIMIZATION objectives -> Higher normalized is BETTER
                # Social Cohesion is MAXIMIZATION objective -> Higher normalized is BETTER
                if key in ['academic_variance', 'wellbeing_risk_variance']:
                    # Scale to 0-1 where 0 is max variance and 1 is min variance (better)
                    normalized_values[key] = 1.0 - ((value - min_val) / (max_val - min_val))
                elif key == 'social_cohesion':
                    # Scale to 0-1 where 0 is min social cohesion and 1 is max (better)
                    normalized_values[key] = (value - min_val) / (max_val - min_val)
                else:
                    # Default to direct normalization if objective type is unknown (shouldn't happen with current objectives)
                     normalized_values[key] = (value - min_val) / (max_val - min_val)

        normalized_values_list.append(normalized_values)

    return normalized_values_list


def calculate_scalar_fitness(normalized_objective_values, weights):
    """Combines normalized objectives into a single fitness score using weights."""
    fitness = 0
    # The normalized values are already adjusted so higher is always better (closer to 1)
    # for maximizing the scalar fitness.
    for key, weight in weights.items():
         fitness += normalized_objective_values.get(key, 0) * weight
    return fitness


def tournament_selection(population, fitness_scores, num_parents, tournament_size):
    """Selects parents using tournament selection."""
    parents = []
    pop_indices = list(range(len(population)))
    # Ensure tournament size doesn't exceed population size
    current_tournament_size = min(tournament_size, len(pop_indices))
    if current_tournament_size < 1: # Handle very small populations
         return []

    for _ in range(num_parents):
        # Select random individuals for the tournament
        tournament_indices = random.sample(pop_indices, current_tournament_size)
        # Find the winner (highest fitness)
        winner_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        parents.append(population[winner_index])
    return parents


def crossover(parent1_alloc, parent2_alloc, df):
    """Performs crossover (simplified - swaps students between corresponding classes), followed by repair."""
    num_classes = len(parent1_alloc)
    num_students = len(df)

    # Create mappings from student ID to their class index in each parent
    parent1_map = {sid: i for i, cls in enumerate(parent1_alloc) for sid in cls}
    parent2_map = {sid: i for i, cls in enumerate(parent2_alloc) for sid in cls}

    # --- Crossover Step ---
    # Create a temporary list of desired class assignments for the offspring
    temp_assignment = [None] * num_students
    student_ids = df['StudentID'].tolist()

    for i, student_id in enumerate(student_ids):
         p1_class = parent1_map.get(student_id, -1)
         p2_class = parent2_map.get(student_id, -1)

         if p1_class == -1 and p2_class == -1:
              chosen_class = random.randrange(num_classes) # Should not happen with valid parents
         elif p1_class == -1:
             chosen_class = p2_class
         elif p2_class == -1:
             chosen_class = p1_class
         else:
             # Randomly choose which parent's class assignment to inherit (50/50 chance)
             chosen_class = random.choice([p1_class, p2_class])

         temp_assignment[i] = chosen_class

    # --- Repair Step ---
    # Redistribute students to meet target class sizes
    offspring_alloc = [[] for _ in range(num_classes)]
    current_sizes = {i: temp_assignment.count(i) for i in range(num_classes)}
    target_size_base = num_students // num_classes
    target_sizes = {i: target_size_base + (1 if i < (num_students % num_classes) else 0) for i in range(num_classes)}

    # Build initial allocation based on temp_assignment
    for i, student_id in enumerate(student_ids):
         offspring_alloc[temp_assignment[i]].append(student_id)


    # Identify overloaded and underloaded classes
    # Sort to prioritize removing from most overloaded and adding to most underloaded
    overloaded_classes = sorted([i for i, size in current_sizes.items() if size > target_sizes[i]], key=lambda i: current_sizes[i], reverse=True)
    underloaded_classes = sorted([i for i, size in current_sizes.items() if size < target_sizes[i]], key=lambda i: current_sizes[i])

    # Move students from overloaded to underloaded classes
    while overloaded_classes and underloaded_classes:
        over_class_id = overloaded_classes[0]
        under_class_id = underloaded_classes[0]

        # Determine how many students to move (min of excess in overloaded and deficit in underloaded)
        num_to_move = min(current_sizes[over_class_id] - target_sizes[over_class_id],
                          target_sizes[under_class_id] - current_sizes[under_class_id])

        if num_to_move <= 0: # Should not happen if loops are active, but safety check
             break

        # Select random students from the overloaded class to move
        students_in_over_class = offspring_alloc[over_class_id] # List of student IDs
        # Ensure we don't try to sample more students than are available
        students_to_move_ids = random.sample(students_in_over_class, min(num_to_move, len(students_in_over_class)))

        for student_id_to_move in students_to_move_ids:
            # Remove from overloaded class and add to underloaded class
            offspring_alloc[over_class_id].remove(student_id_to_move)
            offspring_alloc[under_class_id].append(student_id_to_move)

            # Update current sizes
            current_sizes[over_class_id] -= 1
            current_sizes[under_class_id] += 1

        # Update overloaded/underloaded lists if classes are now at target size
        if current_sizes[over_class_id] == target_sizes[over_class_id]:
            overloaded_classes.pop(0)
        if current_sizes[under_class_id] == target_sizes[under_class_id]:
            underloaded_classes.pop(0)

    # Note: There might be a slight deficit/excess remaining if the repair logic is imperfect
    # or if num_students is very small relative to num_classes. A robust repair ensures exact sizes.
    # This basic repair should work for reasonable class sizes.

    return offspring_alloc


def mutate(allocation, mutation_rate):
    """Performs mutation (swaps students between random classes) preserving class sizes."""
    num_classes = len(allocation)
    num_students_total = sum(len(cls) for cls in allocation) # Get total students from allocation
    mutated_alloc = [list(cls) for cls in allocation] # Create a deep copy

    # Decide how many *students* to potentially involve in a swap
    # Each swap involves 2 students, so number of swap operations is roughly mutation_rate * num_students_total / 2
    num_swap_operations = int(num_students_total * mutation_rate / 2)

    for _ in range(num_swap_operations):
        # Ensure we have at least two classes to swap between and they are not empty
        valid_classes = [i for i, cls in enumerate(mutated_alloc) if cls]
        if len(valid_classes) < 2:
            break

        # Select two random, non-empty classes
        class1_id, class2_id = random.sample(valid_classes, 2)

        # Select one random student from each chosen class
        student1_id = random.choice(mutated_alloc[class1_id])
        student2_id = random.choice(mutated_alloc[class2_id])

        # Perform the swap
        mutated_alloc[class1_id].remove(student1_id)
        mutated_alloc[class1_id].append(student2_id)

        mutated_alloc[class2_id].remove(student2_id)
        mutated_alloc[class2_id].append(student1_id)

    return mutated_alloc


# --- Main Genetic Algorithm Function ---

def run_genetic_allocation(df):
    """Runs the Genetic Algorithm to find an optimal classroom allocation."""
    print(f"Starting Genetic Algorithm with {NUM_STUDENTS} students, {N_CLASSES} classes...")
    start_time = time.time()

    num_students = len(df)
    num_classes = N_CLASSES # Use N_CLASSES defined globally
    num_elitism = max(1, int(GA_POP_SIZE * GA_ELITISM_RATE)) # Number of individuals for elitism

    # Pre-process data for faster objective evaluation
    student_data_map = df.set_index('StudentID').to_dict(orient='index')
    friends_map = df.set_index('StudentID')['Friends'].to_dict()


    # 1. Population Initialization
    population = []
    num_heuristic = int(GA_POP_SIZE * GA_HEURISTIC_SEED_PERCENT)
    for _ in range(num_heuristic):
        population.append(create_heuristic_allocation(df, num_classes))
    while len(population) < GA_POP_SIZE:
        population.append(create_random_allocation(df, num_classes))

    print(f"Initialized population of {GA_POP_SIZE} allocations ({num_heuristic} heuristic, {GA_POP_SIZE - num_heuristic} random).")

    best_allocation = None
    best_fitness = -float('inf')
    best_objectives = None
    fitness_history = []
    best_fitness_history_for_improvement_check = [] # Store only best fitness for termination check

    # GA Loop
    for generation in range(GA_NUM_GENERATIONS):
        # 2. Fitness Evaluation
        objective_values_pop = [evaluate_objectives(alloc, df, student_data_map, friends_map) for alloc in population]
        normalized_objective_values_pop = normalize_objectives(objective_values_pop)
        fitness_scores = [calculate_scalar_fitness(norm_obj, FITNESS_WEIGHTS) for norm_obj in normalized_objective_values_pop]

        # Identify current best individual
        current_best_index = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[current_best_index]
        current_best_allocation = population[current_best_index]
        current_best_objectives = objective_values_pop[current_best_index]

        # Update overall best
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_allocation = current_best_allocation # Store the allocation structure
            best_objectives = current_best_objectives # Store the raw objectives
            print(f"Generation {generation}: New best fitness = {best_fitness:.4f}")
            obj_str = ", ".join([f"{k}: {v:.4f}" for k, v in best_objectives.items()])
            print(f"  Objectives: {obj_str}")

        fitness_history.append(current_best_fitness)
        best_fitness_history_for_improvement_check.append(best_fitness)


        # --- Adaptive Mutation ---
        # Calculate fitness variance of the current population
        fitness_variance = np.var(fitness_scores) if len(fitness_scores) > 1 else 0.0
        current_mutation_rate = GA_MUTATION_RATE_HIGH if fitness_variance < FITNESS_VARIANCE_THRESHOLD else GA_MUTATION_RATE_LOW
        # print(f"Gen {generation}: Fitness Variance = {fitness_variance:.6f}, Mutation Rate = {current_mutation_rate}")


        # --- Termination Check (Improvement based) ---
        if generation >= IMPROVEMENT_CHECK_GENERATIONS:
             recent_best_fitness = best_fitness_history_for_improvement_check[-IMPROVEMENT_CHECK_GENERATIONS:]
             improvement = best_fitness - min(recent_best_fitness) # Compare current best to best from N gens ago
             # print(f"Gen {generation}: Improvement over last {IMPROVEMENT_CHECK_GENERATIONS} gens = {improvement:.6f}")
             if improvement < FITNESS_IMPROVEMENT_THRESHOLD:
                 print(f"Termination condition met: Improvement below {FITNESS_IMPROVEMENT_THRESHOLD} over {IMPROVEMENT_CHECK_GENERATIONS} generations.")
                 break # Stop the GA loop


        # --- Selection and Reproduction ---
        # Elitism: Select the top individuals to carry over directly
        top_indices = np.argsort(fitness_scores)[-num_elitism:][::-1]
        next_population = [population[i] for i in top_indices]

        # Select parents for crossover
        parents = tournament_selection(population, fitness_scores, GA_POP_SIZE, GA_TOURNAMENT_SIZE)

        # Generate offspring to fill the rest of the population
        offspring = []
        while len(next_population) + len(offspring) < GA_POP_SIZE:
             # Select two parents (can be the same) from the selected parent pool
             p1, p2 = random.sample(parents, 2)
             child = crossover(p1, p2, df)
             mutated_child = mutate(child, current_mutation_rate) # Use adaptive mutation rate
             offspring.append(mutated_child)

        next_population.extend(offspring)
        population = next_population # Replace old population


    end_time = time.time()
    duration = end_time - start_time
    print(f"Genetic Algorithm finished after {generation + 1} generations in {duration:.2f} seconds.")
    print(f"Best fitness found: {best_fitness:.4f}")
    print("Best Allocation Objectives:")
    if best_objectives:
        for k, v in best_objectives.items():
            print(f"  {k}: {v:.4f}")

    # Convert the best allocation (list of lists) to a DataFrame with 'Allocated_Class'
    allocated_student_list = []
    # Ensure best_allocation is not None (could happen if GA loop breaks immediately or pop size is too small)
    if best_allocation is not None:
        for class_id, student_list in enumerate(best_allocation):
            for student_id in student_list:
                allocated_student_list.append({'StudentID': student_id, 'Allocated_Class': class_id})
    else:
         print("Warning: No valid allocation found by GA. Returning empty allocation.")
         # Create a dummy allocation
         allocated_student_list = [{'StudentID': sid, 'Allocated_Class': 0} for sid in df['StudentID']] # Assign all to class 0
         best_objectives = {} # Empty objectives
         best_allocation = [df['StudentID'].tolist()] # All students in one class

    df_allocated = pd.DataFrame(allocated_student_list)

    # Merge with the original df to get all student data including scores/flags
    df_final = pd.merge(df, df_allocated, on='StudentID', how='left')

    # --- Step 6: Validation and Reporting for the Best Allocation ---
    allocation_validation_metrics = {}

    # Calculate metrics per class using the final df with allocation
    # Use the actual number of classes in the best allocation
    actual_num_classes = len(best_allocation)
    for class_id in range(actual_num_classes):
        students_in_class = df_final[df_final['Allocated_Class'] == class_id]
        if students_in_class.empty:
             allocation_validation_metrics[f"Class {class_id}"] = {
                 "Size": 0, "Avg Academic Perf": np.nan, "Avg Wellbeing Risk": np.nan,
                 "Avg Peer Score": np.nan, "ESL %": np.nan, "Bully Count": 0,
                 "Vulnerable Count": 0, "Bully-Vulnerable Conflicts": "No", "Multiple Bullies": "No",
                 "Intra-Class Friends": 0
             }
             continue

        avg_academic = students_in_class['Academic_Performance'].mean()
        avg_wellbeing_risk = students_in_class['Wellbeing_Risk'].mean()
        avg_peer_score = students_in_class['Peer_Score'].mean()
        esl_percentage = (students_in_class['language'] == 1).mean() * 100
        bully_count = students_in_class['Is_Bully'].sum()
        vulnerable_count = students_in_class['Is_Vulnerable'].sum()

        has_bully = bully_count > 0
        has_vulnerable = vulnerable_count > 0
        bully_vulnerable_conflict = has_bully and has_vulnerable

        multiple_bullies = bully_count > 1

        # Recalculate intra-class friendships specifically for this class
        class_intra_friendships = 0
        class_student_ids = students_in_class['StudentID'].tolist()
        class_friends_map_subset = students_in_class.set_index('StudentID')['Friends'].to_dict()

        for student_id in class_student_ids:
            friends_str = class_friends_map_subset.get(student_id, "")
            if friends_str and isinstance(friends_str, str): # Ensure it's a string before splitting
                friends_list = [f.strip() for f in friends_str.split(',') if f.strip()]
                for friend_id in friends_list:
                    if friend_id in class_student_ids: # Check if friend is in the *same* class
                         class_intra_friendships += 0.5 # Count each friendship once (A->B and B->A)


        allocation_validation_metrics[f"Class {class_id}"] = {
            "Size": len(students_in_class),
            "Avg Academic Perf": round(avg_academic, 2),
            "Avg Wellbeing Risk": round(avg_wellbeing_risk, 3),
            "Avg Peer Score": round(avg_peer_score, 3),
            "ESL %": round(esl_percentage, 2),
            "Bully Count": int(bully_count),
            "Vulnerable Count": int(vulnerable_count),
            "Bully-Vulnerable Conflicts (Class)": "Yes" if bully_vulnerable_conflict else "No", # Renamed for clarity
            "Multiple Bullies (Class)": "Yes" if multiple_bullies else "No", # Renamed for clarity
            "Intra-Class Friends": int(class_intra_friendships)
        }

    validation_df = pd.DataFrame.from_dict(allocation_validation_metrics, orient='index')
    validation_df.index.name = 'Class'
    validation_df = validation_df.reset_index()


    print("Genetic Algorithm allocation and validation complete.")
    return df_final, validation_df, best_objectives # Return final df, validation report, and overall GA objectives


# --- Gradio Interface Helper Functions ---
def plot_network(df_class, student_id_col='StudentID', friend_col='Friends', color_metric='Academic_Performance'):
    """Generates a NetworkX graph visualization for a class."""
    if df_class.empty:
        return None

    G = nx.Graph()
    all_students_in_class = df_class[student_id_col].tolist()
    G.add_nodes_from(all_students_in_class)

    # Map student IDs to their metric value for coloring
    if color_metric not in df_class.columns:
        # Fallback if metric is missing
        print(f"Warning: Color metric '{color_metric}' not found in class data. Using default color.")
        node_colors = 'skyblue' # Default color
        min_val, max_val = 0, 1 # Dummy range for colorbar
        cmap = cm.viridis # Keep default colormap
        norm = plt.Normalize(vmin=0, vmax=1)
        is_colored = False
    else:
        metric_values = df_class.set_index(student_id_col)[color_metric]
         # Handle potential non-numeric or NaN/inf values in the metric
        metric_values = pd.to_numeric(metric_values, errors='coerce').fillna(metric_values.median() if not metric_values.empty and not metric_values.median() is np.nan else 0)
        metric_values = metric_values.replace([np.inf, -np.inf], metric_values.median() if not metric_values.empty and not metric_values.median() is np.nan else 0)


        min_val, max_val = metric_values.min(), metric_values.max()
        cmap = cm.viridis
        if min_val == max_val:
            norm = plt.Normalize(vmin=min_val - 0.1, vmax=max_val + 0.1) # Add some range if constant
        else:
            norm = plt.Normalize(vmin=min_val, vmax=max_val)
        node_colors = [cmap(norm(metric_values.get(node, norm.vmin))) for node in G.nodes()]
        is_colored = True

    # Add edges based on the 'Friends' column (within the class)
    for index, row in df_class.iterrows():
        student = row[student_id_col]
        friends_str = row[friend_col]
        if pd.isna(friends_str) or not friends_str:
            continue
        if not isinstance(friends_str, str): # Ensure friends_str is string
             friends_str = str(friends_str)
        friends_list = [f.strip() for f in friends_str.split(',') if f.strip()]
        for friend in friends_list:
            # Add edge only if the friend is also in the *same class*
            if friend in all_students_in_class:
                G.add_edge(student, friend)

    fig, ax = plt.subplots(figsize=(10, 8))

    try:
        graph_size = len(G.nodes())
        k_value = 0.5 / np.sqrt(graph_size) if graph_size > 0 else 0.1
        pos = nx.spring_layout(G, k=k_value, iterations=50, seed=42)
    except Exception as e:
         print(f"Error generating layout, using random_layout: {e}")
         pos = nx.random_layout(G, seed=42)

    nx.draw(G, pos, ax=ax,
            node_size=50,
            width=0.5,
            with_labels=False,
            node_color=node_colors) # Use the calculated node_colors

    ax.set_title(f"Friendship Network within Class (Colored by {color_metric if is_colored else 'Default'})")

    # Add a colorbar only if metric coloring was applied
    if is_colored:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label(color_metric)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

def plot_histogram(df_class, metric, class_id):
    """Generates a histogram for a given metric within a class."""
    if df_class.empty or metric not in df_class.columns:
        return None

    # Handle potential non-numeric data by coercing to numeric and dropping NaNs
    data = pd.to_numeric(df_class[metric], errors='coerce').dropna()

    if data.empty:
         return None

    fig, ax = plt.subplots(figsize=(6, 4))
    # Ensure bins are positive integer, max(5, min(20, len(data)//10)) can be 0 or less for small data
    bins = max(5, min(20, int(len(data)/10))) if len(data) > 10 else min(max(1, len(data)//2), 5) # More robust bin calculation

    ax.hist(data, bins=bins, alpha=0.7)
    ax.set_title(f"{metric} Distribution in Class {class_id}")
    ax.set_xlabel(metric)
    ax.set_ylabel("Frequency")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


# --- Data Loading and Processing (Global scope) ---
# This runs once when the script starts
df_processed = None
allocation_validation_report = None
best_ga_objectives = None
ga_classes = []

# Define available metrics for coloring networks and histograms
all_available_metrics = ['Academic_Performance', 'Academic_Risk', 'Wellbeing_Risk',
                          'Peer_Score', 'Friends_Count', 'k6_overall', 'criticises',
                          'Is_Bully', 'Is_Vulnerable', 'Is_Supportive', 'language',
                          'GrowthMindset', 'Manbox5_overall', 'Masculinity_contrained',
                          'School_support_engage'] # Include all potentially useful metrics


try:
    df_synthetic = generate_synthetic_data()
    # Run predictive analysis to get risk scores, bullying/vulnerable flags, etc.
    # These are used as inputs for the GA objective functions.
    df_processed = run_predictive_analysis(df_synthetic.copy())

    # Run the Genetic Algorithm allocation
    df_processed, allocation_validation_report, best_ga_objectives = run_genetic_allocation(df_processed)

    # Get unique class labels for dropdowns from the GA result
    if 'Allocated_Class' in df_processed.columns:
        ga_classes = sorted(df_processed['Allocated_Class'].unique().tolist())

    all_classes = ga_classes # Only GA allocation available now

    # Update available metrics based on the final DataFrame columns
    available_color_metrics = [metric for metric in all_available_metrics if metric in df_processed.columns]

except Exception as e:
    print(f"FATAL ERROR during initial data processing or GA allocation: {e}")
    import traceback
    traceback.print_exc() # Print traceback for debugging
    # Create dummy data and classes to prevent Gradio from crashing on launch
    df_processed = pd.DataFrame({'StudentID': [f'ErrorS{i}' for i in range(N_CLASSES * 25)], 'Allocated_Class': [i % N_CLASSES for i in range(N_CLASSES * 25)]})
    # Add dummy essential columns that plot_network/histogram might need
    for col in all_available_metrics + ['Friends', 'criticises', 'k6_overall', 'language', 'Is_Bully', 'Is_Vulnerable']:
         if col not in df_processed.columns:
              df_processed[col] = 0
    df_processed['StudentID'] = df_processed['StudentID'].astype(str) # Ensure StudentID is string
    df_processed['Friends'] = df_processed['StudentID'].apply(lambda x: f"ErrorS{random.randint(0, N_CLASSES*25-1)}") # Dummy friends

    all_classes = sorted(df_processed['Allocated_Class'].unique().tolist())
    ga_classes = all_classes
    allocation_validation_report = pd.DataFrame({
        'Class': [f'Error {i}' for i in range(N_CLASSES)], 'Size': [0] * N_CLASSES, 'Avg Academic Perf': [np.nan] * N_CLASSES,
        'Avg Wellbeing Risk': [np.nan] * N_CLASSES, 'Avg Peer Score': [np.nan] * N_CLASSES,
        'ESL %': [np.nan] * N_CLASSES, 'Bully Count': [0] * N_CLASSES, 'Vulnerable Count': [0] * N_CLASSES,
        'Bully-Vulnerable Conflicts (Class)': ['Error'] * N_CLASSES, 'Multiple Bullies (Class)': ['Error'] * N_CLASSES,
        'Intra-Class Friends': [0] * N_CLASSES
    })
    best_ga_objectives = {"Academic Equity": np.nan, "Wellbeing Balance": np.nan, "Social Cohesion": np.nan}
    available_color_metrics = ['Academic_Performance'] # Fallback


# --- Gradio Interface Update Function ---
def update_visualizations(selected_class_id_str, color_metric):
    """Updates dashboard outputs based on user selections (using GA allocation)."""
    global df_processed, allocation_validation_report, best_ga_objectives # Access global dataframes

    # Check for fatal error state
    if df_processed is None or 'StudentID' not in df_processed or df_processed['StudentID'].iloc[0].startswith('Error'):
         error_message = "Fatal Error during data loading or GA allocation. Cannot display results."
         dummy_df = pd.DataFrame()
         dummy_report = allocation_validation_report if allocation_validation_report is not None else pd.DataFrame()
         dummy_objectives = best_ga_objectives if best_ga_objectives is not None else {}
         obj_html = "### Overall GA Objective Scores<br>Data not available due to error."

         return (dummy_report, dummy_df, error_message, obj_html,
                 "Histogram not available due to error.", "Histogram not available due to error.",
                 "Histogram not available due to error.", "Histogram not available due to error.")


    class_col = 'Allocated_Class'

    # Ensure selected_class_id is an integer
    try:
        selected_class_id = int(selected_class_id_str)
    except (ValueError, TypeError):
        valid_classes = sorted(df_processed[class_col].unique().tolist())
        if not valid_classes:
             return (pd.DataFrame(), pd.DataFrame(), "No classes found for this method.", "No objectives available.",
                     None, None, None, None)
        selected_class_id = valid_classes[0] # Default to the first valid class


    # --- Display Overall GA Objectives ---
    objective_summary_html = "### Overall GA Objective Scores<br>"
    if best_ga_objectives:
         # Map objective keys to user-friendly names and format
         display_names = {
             'academic_variance': 'Academic Equity (Variance)',
             'wellbeing_risk_variance': 'Wellbeing Balance (Risk Variance)',
             'social_cohesion': 'Social Cohesion ((Friends - Conflicts) / N)'
         }
         obj_details = []
         # Display objectives in a consistent order based on weights or a defined list
         ordered_keys = ['academic_variance', 'wellbeing_risk_variance', 'social_cohesion']
         for key in ordered_keys:
              if key in best_ga_objectives:
                  display_name = display_names.get(key, key.replace('_', ' ').title())
                  value = best_ga_objectives[key]
                  # Add explanation of direction (minimize/maximize)
                  direction = "(Minimize)" if key in ['academic_variance', 'wellbeing_risk_variance'] else "(Maximize)"
                  obj_details.append(f"&bull; {display_name}: {value:.4f} {direction}")

         objective_summary_html += "<br>".join(obj_details)
    else:
         objective_summary_html += "GA objectives not available."


    # --- 1. Class Overview / Validation Report Table ---
    # Use the pre-calculated validation report
    overview = allocation_validation_report


    # --- 2. Selected Class Details ---
    df_selected_class = df_processed[df_processed[class_col] == selected_class_id].copy()

    # Select relevant columns for the student table
    student_table_cols = [
        'StudentID', 'Academic_Performance', 'Academic_Risk',
        'Wellbeing_Risk', 'Peer_Score', 'Friends_Count', 'k6_overall',
        'criticises', 'language', 'Is_Bully', 'Is_Vulnerable', 'Is_Supportive',
        'School_support_engage'
    ]
    # Add color metric if not already included and exists
    if color_metric not in student_table_cols and color_metric in df_selected_class.columns:
         student_table_cols.append(color_metric)

    # Ensure all columns exist before selecting
    student_table_cols = [col for col in student_table_cols if col in df_selected_class.columns]
    student_details_df = df_selected_class[student_table_cols]
    # Round numeric columns for display
    for col in student_details_df.columns:
        if pd.api.types.is_numeric_dtype(student_details_df[col]):
            student_details_df[col] = student_details_df[col].round(3)
        # Convert boolean flags to Yes/No or 1/0 if needed for clarity
        if student_details_df[col].dtype == 'int64' and student_details_df[col].isin([0, 1]).all():
             if col.startswith('Is_'):
                  student_details_df[col] = student_details_df[col].map({0: 'No', 1: 'Yes'})


    # --- 3. Network Plot ---
    # Pass only the required columns to avoid issues in plotting function
    df_plot = df_selected_class[[c for c in ['StudentID', 'Friends', color_metric] if c in df_selected_class.columns]]
    network_img_html = "Plot not generated."
    if not df_plot.empty and color_metric in df_plot.columns:
        network_plot_b64 = plot_network(df_plot, color_metric=color_metric)
        if network_plot_b64:
            network_img_html = f'<img src="{network_plot_b64}" alt="Class Network Graph" style="max-width: 100%; height: auto;">'
        else:
            network_img_html = "Could not generate network plot (maybe no students/friends in class or invalid metric)."
    else:
         network_img_html = f"Cannot generate plot: Class {selected_class_id} is empty or required data for plot missing."


    # --- 4. Histogram Plots ---
    # Choose relevant metrics for histograms
    hist_metrics = ['Academic_Performance', 'Wellbeing_Risk', 'Bullying_Score', 'Friends_Count', 'Peer_Score', 'k6_overall']
    hist_html_outputs = {}

    for metric in hist_metrics:
         hist_b64 = plot_histogram(df_selected_class, metric, selected_class_id)
         hist_html_outputs[f'{metric.lower().replace(" ", "_")}_hist_html'] = f'<img src="{hist_b64}" alt="{metric} Distribution" style="max-width: 100%; height: auto;">' if hist_b64 else f"Histogram for {metric} not available."


    return (overview, student_details_df, network_img_html, objective_summary_html,
            hist_html_outputs.get('academic_performance_hist_html', "Histogram not available."),
            hist_html_outputs.get('wellbeing_risk_hist_html', "Histogram not available."),
            hist_html_outputs.get('bullying_score_hist_html', "Histogram not available."),
            hist_html_outputs.get('peer_score_hist_html', "Histogram not available."),
           )

# --- Define Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="GA Classroom Allocation Visualizer") as demo:
    gr.Markdown("# Genetic Algorithm-Based Classroom Allocation Visualization")
    gr.Markdown("View the results of the multi-objective Genetic Algorithm allocation.")

    with gr.Row():
        # Only GA allocation available now
        class_id_dd = gr.Dropdown(
            label="Select Class ID",
            choices=ga_classes if ga_classes else [0],
            value=ga_classes[0] if ga_classes else None,
            interactive=True
        )
        color_metric_dd = gr.Dropdown(
            label="Color Network By",
            choices=available_color_metrics,
            value='Academic_Performance' if 'Academic_Performance' in available_color_metrics else available_color_metrics[0] if available_color_metrics else None,
            interactive=True
        )

    gr.Markdown("## Allocation Objectives Summary")
    # Display the overall objective scores found by the GA
    ga_objectives_summary = gr.HTML(label="Overall Genetic Algorithm Objectives")


    gr.Markdown("## Class Overview / Validation Report")
    # This table shows the validation report for the GA allocation
    overview_table = gr.DataFrame(label=f"Validation Report per Class (Target Total Classes: {N_CLASSES})")

    gr.Markdown("## Selected Class Details")
    with gr.Row():
        with gr.Column(scale=2):
            student_table = gr.DataFrame(label=f"Students in Selected Class (Target Size: {CLASS_SIZE_TARGET})")
        with gr.Column(scale=3):
            gr.Markdown("### Friendship Network")
            network_plot_html = gr.HTML(label="Class Network Graph")

    gr.Markdown("## Metric Distributions for Selected Class")
    # Display histograms for key metrics
    with gr.Row():
        academic_perf_hist_html = gr.HTML(label="Academic Performance Distribution")
        wellbeing_risk_hist_html = gr.HTML(label="Wellbeing Risk Distribution")
    with gr.Row():
         bullying_score_hist_html = gr.HTML(label="Bullying Score Distribution")
         peer_score_hist_html = gr.HTML(label="Peer Score Distribution")


    # Define update triggers
    inputs = [class_id_dd, color_metric_dd]
    outputs = [overview_table, student_table, network_plot_html, ga_objectives_summary,
               academic_perf_hist_html, wellbeing_risk_hist_html, bullying_score_hist_html,
               peer_score_hist_html]

    # Trigger main visualization update when class ID or color metric changes
    class_id_dd.change(
        update_visualizations,
        inputs=inputs,
        outputs=outputs
    )
    color_metric_dd.change(
        update_visualizations,
        inputs=inputs,
        outputs=outputs
    )

    # Load initial data when the app starts
    demo.load(
        update_visualizations,
        inputs=[class_id_dd, color_metric_dd], # Pass default initial values
        outputs=outputs
    )


# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio App...")
    # Share=True creates a public link (optional)
    demo.launch(share=False) # Specify host and port if needed