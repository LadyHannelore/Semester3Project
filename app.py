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
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
import warnings
import os

# --- Configuration & Setup ---
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
plt.style.use('seaborn-v0_8_whitegrid')

NUM_STUDENTS = 1000
CLASS_SIZE_TARGET = 30
N_CLUSTERS = max(1, NUM_STUDENTS // CLASS_SIZE_TARGET)
SYNTHETIC_DATA_CSV = "synthetic_student_data.csv"

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
        academic_performance = max(0, min(100, round(np.random.normal(70, 15))))
        manbox5_scores = {f"Manbox5_{i}": random.choice(LIKERT_SCALE_1_7) for i in range(1, 6)}
        k6_scores = {f"k6_{i}": random.choice(K6_SCALE_1_5) for i in range(1, 7)}
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
            "Friends": ", ".join(random.sample([pid for pid in student_ids if pid != student_id], k=random.randint(0, 7))),
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

# --- Predictive Analytics & Clustering ---
def run_analysis(df):
    """Performs prediction and clustering on the dataframe."""
    print("Running predictive analytics and clustering...")
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

    cluster_features = features + ['Academic_Risk', 'Wellbeing_Risk', 'Peer_Score', 'Friends_Count']
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[cluster_features].fillna(0))

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df['Class_KMeans'] = kmeans.fit_predict(X_cluster)

    spectral = SpectralClustering(n_clusters=N_CLUSTERS, affinity='rbf', random_state=42, n_init=10, assign_labels='kmeans')
    try:
        df['Class_Spectral'] = spectral.fit_predict(X_cluster)
    except Exception as e:
        print(f"Error during Spectral Clustering: {e}")
        df['Class_Spectral'] = df['Class_KMeans']

    print("Analysis complete.")
    return df

# --- Visualization Functions ---
def plot_network(df_class, student_id_col='StudentID', friend_col='Friends', color_metric='Academic_Risk'):
    """Generates a NetworkX graph visualization for a class."""
    if df_class.empty:
        return None # Return None if no data

    G = nx.Graph()
    all_students_in_class = df_class[student_id_col].tolist()
    G.add_nodes_from(all_students_in_class)

    # Map student IDs to their metric value for coloring
    # Use a perceptually uniform colormap like 'viridis' or 'plasma'
    if color_metric not in df_class.columns:
         print(f"Warning: Color metric '{color_metric}' not found in class data. Using default color.")
         # Use a uniform color if the metric is missing
         node_colors = {node: 'skyblue' for node in G.nodes()}
    else:
         metric_values = df_class.set_index(student_id_col)[color_metric]
         # Handle case where all values are the same (min_val == max_val)
         min_val, max_val = metric_values.min(), metric_values.max()
         cmap = cm.viridis # Choose colormap
         node_colors = {}
         for node in G.nodes():
             value = metric_values.get(node, min_val) # Default to min_val if not found
             # Normalize value to 0-1 for colormap, handle division by zero
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
            # Add edge only if the friend is also in the *same class*
            if friend in all_students_in_class:
                G.add_edge(student, friend)

    fig, ax = plt.subplots(figsize=(10, 8)) # Increase figure size

    # Use a layout that spreads nodes out more
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42) # Adjust k for spacing

    nx.draw(G, pos, ax=ax,
            node_size=50,       # Smaller nodes
            width=0.5,          # Thinner edges
            with_labels=False,  # No labels to avoid clutter
            node_color=[node_colors.get(node, 'skyblue') for node in G.nodes()]) # Apply colors

    ax.set_title(f"Friendship Network within Class (Colored by {color_metric})")

    # Add a colorbar only if the metric was found and varied
    if color_metric in df_class.columns and (max_val - min_val) > 0:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
        sm.set_array([]) # Required for ScalarMappable
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label(color_metric)


    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # Close the figure to free memory
    return f"data:image/png;base64,{img_str}" # Return base64 string for HTML

def plot_histogram(df_class, metric, class_id):
    """Generates a histogram for a given metric within a class."""
    if df_class.empty or metric not in df_class.columns or df_class[metric].isnull().all():
        return None # Return None if no data, metric missing, or all values are NaN

    fig, ax = plt.subplots(figsize=(6, 4))
    df_class[metric].plot(kind='hist', ax=ax, bins=10, alpha=0.7) # Use 10 bins
    ax.set_title(f"{metric} Distribution in Class {class_id}")
    ax.set_xlabel(metric)
    ax.set_ylabel("Frequency")
    plt.tight_layout()

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # Close the figure to free memory
    return f"data:image/png;base64,{img_str}" # Return base64 string for HTML


# --- Load and Process Data Globally ---
# Ensure data is generated and analyzed once when the app starts
df_processed = None
all_classes = []
initial_load_error = None

try:
    df_synthetic = generate_synthetic_data()
    df_processed = run_analysis(df_synthetic.copy()) # Work on a copy
    # Get unique class labels for dropdowns for all methods
    kmeans_classes = sorted(df_processed['Class_KMeans'].unique())
    spectral_classes = sorted(df_processed['Class_Spectral'].unique())
    optimized_classes = sorted(df_processed['Class_Optimized'].unique())
    # Use unique classes from Optimized for the initial dropdown population
    all_classes = optimized_classes # Default classes in dropdown
except Exception as e:
    print(f"FATAL ERROR during initial data processing: {e}")
    initial_load_error = f"Error during initial data processing: {e}"
    # Create dummy data to prevent Gradio from crashing on launch
    df_processed = pd.DataFrame({'StudentID': ['Error'], 'Class_KMeans': [0], 'Class_Spectral': [0], 'Class_Optimized': [0],
                                   'Academic_Performance': [0], 'Academic_Risk': [0.5], 'Wellbeing_Risk': [0.5],
                                   'Peer_Score': [0.5], 'Friends': [''], 'Friends_Count': [0], 'k6_overall': [0],
                                   'bullying': [1], 'Is_Loner': [1], 'Is_Bully': [0]
                                 })
    all_classes = [0]

# --- Gradio Interface ---
def update_class_dropdown(clustering_method):
    """Returns the list of available class IDs for the selected method."""
    if df_processed is None or 'StudentID' not in df_processed or df_processed['StudentID'].iloc[0] == 'Error':
        return gr.Dropdown(choices=[], value=None, interactive=False) # Return empty dropdown on error

    if clustering_method == 'K-Means':
        class_col = 'Class_KMeans'
    elif clustering_method == 'Spectral':
        class_col = 'Class_Spectral'
    else: # Optimized (K-Means Base)
        class_col = 'Class_Optimized'

    if class_col not in df_processed.columns:
         return gr.Dropdown(choices=[], value=None, interactive=False)

    unique_classes = sorted(df_processed[class_col].unique())
    return gr.Dropdown(choices=unique_classes, value=unique_classes[0] if unique_classes else None, interactive=True)


def update_visualizations(clustering_method, selected_class_id_str, color_metric):
    """Updates dashboard outputs based on user selections."""
    if initial_load_error:
         return pd.DataFrame(), pd.DataFrame(), initial_load_error, initial_load_error, initial_load_error

    if df_processed is None or 'StudentID' not in df_processed or df_processed['StudentID'].iloc[0] == 'Error':
        return pd.DataFrame(), pd.DataFrame(), "Error: Dataframe not loaded correctly.", "Error: Dataframe not loaded correctly.", "Error: Dataframe not loaded correctly."

    class_col = 'Class_KMeans' if clustering_method == 'K-Means' else ('Class_Spectral' if clustering_method == 'Spectral' else 'Class_Optimized')

    # Ensure selected_class_id is valid for the chosen method
    try:
        selected_class_id = int(selected_class_id_str)
        if selected_class_id not in df_processed[class_col].unique():
            raise ValueError("Selected Class ID not found for this method.")
    except (ValueError, TypeError) as e:
        print(f"Invalid class selection: {e}. Attempting to default to first available class.")
        valid_classes = sorted(df_processed[class_col].unique())
        if not valid_classes:
            return pd.DataFrame(), pd.DataFrame(), "No classes found for this method.", "No classes found for this method.", "No classes found for this method."
        selected_class_id = valid_classes[0]
        # Also update the dropdown value in the UI to the defaulted ID - This is tricky within a single function.
        # The dropdown's value will only update on the next trigger.

    # --- 1. Class Overview Table ---
    overview = df_processed.groupby(class_col).agg(
        Class_Size=('StudentID', 'count'),
        Avg_Academic_Perf=('Academic_Performance', 'mean'),
        Avg_Academic_Risk=('Academic_Risk', 'mean'),
        Avg_Wellbeing_Risk=('Wellbeing_Risk', 'mean'),
        Avg_Peer_Score=('Peer_Score', 'mean'),
        Avg_Friends_Count=('Friends_Count', 'mean'),
        Bully_Count=('Is_Bully', 'sum'), # Add bully count
        Loner_Count=('Is_Loner', 'sum') # Add loner count
    ).reset_index()
    overview.rename(columns={class_col: 'Class_ID'}, inplace=True) # Rename the class ID column for clarity
    overview = overview.round(2) # Round for display

    # --- 2. Selected Class Details ---
    df_selected_class = df_processed[df_processed[class_col] == selected_class_id].copy()

    # Select relevant columns for the student table
    student_table_cols = [
        'StudentID', 'Academic_Performance', 'Academic_Risk',
        'Wellbeing_Risk', 'Peer_Score', 'Friends_Count', 'k6_overall',
        'bullying', 'Is_Loner', 'Is_Bully' # Add bully/loner flags
    ]
    # Add color metric if not already included and if it exists in the dataframe
    if color_metric not in student_table_cols and color_metric in df_processed.columns:
        student_table_cols.append(color_metric)

    # Ensure all columns exist before selecting
    student_table_cols = [col for col in student_table_cols if col in df_selected_class.columns]
    student_details_df = df_selected_class[student_table_cols].round(3)

    # --- 3. Network Plot ---
    print(f"Generating network plot for class {selected_class_id} using {clustering_method} results, colored by {color_metric}...")
    network_img_html = None
    # Check if color metric exists in the selected class data before attempting to plot
    if not df_selected_class.empty and color_metric in df_selected_class.columns:
        network_plot_b64 = plot_network(df_selected_class, color_metric=color_metric)
        if network_plot_b64:
             network_img_html = f'<img src="{network_plot_b64}" alt="Class Network Graph" style="max-width: 100%; height: auto;">'
        else:
             network_img_html = "Could not generate network plot (maybe no friends within class or invalid metric)."
    else:
        network_img_html = f"Cannot generate plot: Class {selected_class_id} is empty or color metric '{color_metric}' not found."


    # --- 4. Histogram Plots ---
    print(f"Generating histograms for class {selected_class_id}...")
    # Check if metrics exist before plotting
    hist_academic_html = "Academic_Risk data not available."
    if 'Academic_Risk' in df_selected_class.columns and not df_selected_class['Academic_Risk'].isnull().all():
        hist_academic_b64 = plot_histogram(df_selected_class, 'Academic_Risk', selected_class_id)
        hist_academic_html = f'<img src="{hist_academic_b64}" alt="Academic Risk Distribution">' if hist_academic_b64 else "Could not generate Academic Risk histogram."

    hist_wellbeing_html = "Wellbeing_Risk data not available."
    if 'Wellbeing_Risk' in df_selected_class.columns and not df_selected_class['Wellbeing_Risk'].isnull().all():
        hist_wellbeing_b64 = plot_histogram(df_selected_class, 'Wellbeing_Risk', selected_class_id)
        hist_wellbeing_html = f'<img src="{hist_wellbeing_b64}" alt="Wellbeing Risk Distribution">' if hist_wellbeing_b64 else "Could not generate Wellbeing Risk histogram."


    return overview, student_details_df, network_img_html, hist_academic_html, hist_wellbeing_html


# --- Define Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="Classroom Allocation Visualizer") as demo:
    gr.Markdown("# AI-Powered Classroom Allocation Visualization")
    gr.Markdown("Explore the results of predictive modeling and clustering for student classroom allocation. **The 'Optimized' method attempts to distribute potential 'bullies' and place 'loners' after initial clustering based on K-Means.**")

    if initial_load_error:
        gr.Warning(initial_load_error)
    else:
        gr.Info("Data loaded and processed successfully.")

    with gr.Row():
        clustering_method_dd = gr.Dropdown(
            label="Clustering Method",
            choices=['K-Means', 'Spectral', 'Optimized (K-Means Base)'], # Add Optimized option
            value='Optimized (K-Means Base)', # Default value to show optimized results first
            interactive=True
        )
        color_metric_dd = gr.Dropdown(
             label="Color Network By",
             choices=['Academic_Risk', 'Wellbeing_Risk', 'Peer_Score', 'Academic_Performance', 'k6_overall', 'Friends_Count', 'bullying'], # Add bullying as color option
             value='Academic_Risk',
             interactive=True
        )
        apply_btn = gr.Button("Apply Selections")


    with gr.Row(): # Move class_id_dd here or keep separate? Let's keep separate for clarity
        class_id_dd = gr.Dropdown(
            label="Select Class ID",
            choices=all_classes, # Initial choices populated from Optimized results
            value=all_classes[0] if all_classes else None,
            interactive=True
        )


    gr.Markdown("## Class Overview")
    gr.Markdown("_(Note: 'Optimized' method aims for better distribution/placement for specific groups but may not align perfectly with initial clustering metrics)_")
    overview_table = gr.DataFrame(label="Summary statistics for all classes")

    gr.Markdown("## Selected Class Details")
    with gr.Row():
        with gr.Column(scale=2):
            student_table = gr.DataFrame(label="Students in Selected Class")
        with gr.Column(scale=3):
            gr.Markdown("### Friendship Network")
            network_plot_html = gr.HTML(label="Class Network Graph")

    gr.Markdown("### Metric Distributions for Selected Class")
    with gr.Row():
        academic_hist_html = gr.HTML(label="Academic Risk Distribution")
        wellbeing_hist_html = gr.HTML(label="Wellbeing Risk Distribution")


    # Define update triggers
    # Update class ID dropdown when clustering method changes
    clustering_method_dd.change(
        update_class_dropdown,
        inputs=[clustering_method_dd],
        outputs=[class_id_dd]
    )

    # Update visualizations when the button is clicked
    apply_btn.click(
        update_visualizations,
        inputs=[clustering_method_dd, class_id_dd, color_metric_dd],
        outputs=[overview_table, student_table, network_plot_html, academic_hist_html, wellbeing_hist_html]
    )

    # Initial load: Update class dropdown based on default method, then trigger visualization update using default selections
    demo.load(
        update_class_dropdown,
        inputs=[clustering_method_dd], # Use default value of clustering_method_dd on load
        outputs=[class_id_dd]
    ).then(
        update_visualizations,
        inputs=[clustering_method_dd, class_id_dd, color_metric_dd], # Use current values of dropdowns
        outputs=[overview_table, student_table, network_plot_html, academic_hist_html, wellbeing_hist_html]
    )


# --- Main Execution ---
if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch(share=False)