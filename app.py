import gradio as gr
import streamlit as st
import pandas as pd
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Import colormap functionality
import io  # To handle plot image bytes
import base64  # To encode plot image for HTML display
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
import warnings
import os  # To check if data file exists
SYNTHETIC_DATA_CSV="synthetic_student_data.csv"
NUM_STUDENTS = 10000 # Number of students to generate  
# --- Configuration & Setup ---
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')  # Suppress KMeans memory leak warning on Windows
plt.style.use('seaborn-v0_8-whitegrid')  # Use a clean plot style

st.title("Classroom Allocation App")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    # AI Allocation Priority Selection
    allocation_priority = st.selectbox(
        "Select AI Allocation Priority",
        ["Balancing Academic Performance", "Maximizing Collaboration", "Minimizing Disruption"],
        index=0,
    )

    # Customization Panel
    st.header("Customization Panel")
    academic_weight = st.slider("Academic Performance Weight", 0.0, 1.0, 0.5)
    wellbeing_weight = st.slider("Wellbeing Indicator Weight", 0.0, 1.0, 0.5)
    disruption_weight = st.slider("Disruptive Behavior Weight", 0.0, 1.0, 0.5)
    social_weight = st.slider("Social Network Centrality Weight", 0.0, 1.0, 0.5)

# --- Data Generation (Slightly modified from previous example) ---
def generate_synthetic_data(filename=SYNTHETIC_DATA_CSV, num_students=NUM_STUDENTS):
    """Generates synthetic student data if the CSV doesn't exist."""
    if os.path.exists(filename):
        print(f"Loading existing data from {filename}")
        try:
            df = pd.read_csv(filename)
            if len(df) == num_students:
                 # Basic check if required columns exist from previous generation
                required_cols = ['StudentID', 'Academic_Performance', 'Friends']
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
    # Define possible scales based on survey info
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
        # Simplified generation for brevity in this example
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

    # --- Calculate Derived Fields ---
    df['Manbox5_overall'] = df[[f"Manbox5_{i}" for i in range(1, 6)]].mean(axis=1)
    df['Masculinity_contrained'] = df[['Soft', 'WomenDifferent', 'Nerds', 'MenBetterSTEM']].mean(axis=1)
    df['GrowthMindset'] = ((8.0 - df['Intelligence1']) + (8.0 - df['Intelligence2'])) / 2.0
    df['k6_overall'] = df[[f"k6_{i}" for i in range(1, 7)]].sum(axis=1)
    df['School_support_engage6'] = ((8.0 - df['isolated']) + (8.0 - df['opinion']) + df['criticises'] + df['comfortable'] + df['bullying'] + df['future']) / 6.0
    df['School_support_engage'] = df[['criticises', 'comfortable', 'bullying', 'future']].mean(axis=1)

    df.to_csv(filename, index=False)
    print(f"Synthetic data saved to {filename}")
    return df

def run_analysis(df):
    """Performs prediction and clustering on the dataframe."""
    print("Running predictive analytics and clustering...")
    # 1. Simulate Labels
    df['Academic_Success'] = (df['Academic_Performance'] > df['Academic_Performance'].quantile(0.75)).astype(int)
    df['Wellbeing_Decline'] = (df['k6_overall'] > df['k6_overall'].quantile(0.75)).astype(int)
    # Handle potential NaN/empty strings in 'Friends' before splitting
    df['Friends_Count'] = df['Friends'].fillna('').apply(lambda x: len(x.split(', ')) if x else 0)
    df['Positive_Peer_Collab'] = (df['Friends_Count'] > df['Friends_Count'].median()).astype(int)

    # 2. Train Models
    features = [
        'Academic_Performance', 'isolated', 'WomenDifferent', 'language',
        'pwi_wellbeing', 'GrowthMindset', 'k6_overall', 'Manbox5_overall',
        'Masculinity_contrained', 'School_support_engage6', 'School_support_engage'
    ]
    # Ensure all features exist and handle potential missing ones (e.g., fill with median)
    for col in features:
        if col not in df.columns:
            print(f"Warning: Feature '{col}' not found. Filling with 0.")
            df[col] = 0
        elif df[col].isnull().any():
            print(f"Warning: Feature '{col}' has NaNs. Filling with median.")
            df[col] = df[col].fillna(df[col].median())

    X = df[features]
    y_academic = df['Academic_Success']
    y_wellbeing = df['Wellbeing_Decline']
    y_peer = df['Positive_Peer_Collab']

    # Simple train split (optional, could train on all data for prediction)
    # X_train, X_test, y_academic_train, y_academic_test = train_test_split(X, y_academic, test_size=0.2, random_state=42)
    # For prediction on the whole dataset:
    X_train, y_academic_train = X, y_academic
    X_train_wb, y_wellbeing_train = X, y_wellbeing
    X_train_p, y_peer_train = X, y_peer

    try:
        academic_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(X_train, y_academic_train)
        wellbeing_model = RandomForestClassifier(random_state=42).fit(X_train_wb, y_wellbeing_train)
        peer_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(X_train_p, y_peer_train)

        # Predict probabilities on the entire dataset
        df['Academic_Risk'] = academic_model.predict_proba(X)[:, 0]  # P(not succeeding)
        df['Wellbeing_Risk'] = wellbeing_model.predict_proba(X)[:, 1]  # P(decline)
        df['Peer_Score'] = peer_model.predict_proba(X)[:, 1]  # P(positive collaboration)
    except Exception as e:
        print(f"Error during model training/prediction: {e}")
        # Assign default risk scores if prediction fails
        df['Academic_Risk'] = 0.5
        df['Wellbeing_Risk'] = 0.5
        df['Peer_Score'] = 0.5

    # 3. Clustering
    cluster_features = features + ['Academic_Risk', 'Wellbeing_Risk', 'Peer_Score']
    # Add network features (simplified)
    df['Degree_Centrality'] = df['Friends_Count']
    cluster_features += ['Degree_Centrality']

    # Ensure cluster features exist and handle NaNs
    for col in cluster_features:
         if col not in df.columns:
            print(f"Warning: Cluster Feature '{col}' not found. Filling with 0.")
            df[col] = 0
         elif df[col].isnull().any():
            print(f"Warning: Cluster Feature '{col}' has NaNs. Filling with median.")
            df[col] = df[col].fillna(df[col].median())

    # Scale features
    scaler = StandardScaler()
    # Ensure no NaN/inf values before scaling
    X_cluster_raw = df[cluster_features].replace([np.inf, -np.inf], np.nan)
    X_cluster_raw = X_cluster_raw.fillna(X_cluster_raw.median()) # Fill any remaining NaNs
    X_cluster = scaler.fit_transform(X_cluster_raw)

    # K-Means
    print(f"Running K-Means with {N_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10) # Explicitly set n_init
    df['Class_KMeans'] = kmeans.fit_predict(X_cluster)

    # Spectral Clustering
    print(f"Running Spectral Clustering with {N_CLUSTERS} clusters...")
    # Use 'rbf' affinity as 'nearest_neighbors' can be slow/memory intensive for 1000 points
    # Reduce n_neighbors if memory issues persist
    spectral = SpectralClustering(
        n_clusters=N_CLUSTERS,
        affinity='rbf', # Changed from nearest_neighbors
        gamma=1.0, # Default gamma, adjust if needed
        random_state=42,
        n_init=10, # Add n_init
        assign_labels='kmeans' # Often more stable
    )
    try:
        df['Class_Spectral'] = spectral.fit_predict(X_cluster)
    except Exception as e:
        print(f"Error during Spectral Clustering: {e}. Assigning K-Means results as fallback.")
        df['Class_Spectral'] = df['Class_KMeans'] # Fallback

    print("Analysis complete.")
    return df

def plot_network(df_class, student_id_col='StudentID', friend_col='Friends', color_metric='Academic_Risk'):
    """Generates a NetworkX graph visualization for a class."""
    if df_class.empty:
        return None # Return None if no data

    G = nx.Graph()
    all_students_in_class = df_class[student_id_col].tolist()
    G.add_nodes_from(all_students_in_class)

    # Map student IDs to their metric value for coloring
    # Use a perceptually uniform colormap like 'viridis' or 'plasma'
    metric_values = df_class.set_index(student_id_col)[color_metric]
    min_val, max_val = metric_values.min(), metric_values.max()
    cmap = cm.viridis # Choose colormap

    node_colors = {}
    for node in G.nodes():
        value = metric_values.get(node, 0) # Default to 0 if not found
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
            # Add edge only if the friend is also in the *same class*
            if friend in all_students_in_class:
                G.add_edge(student, friend)

    fig, ax = plt.subplots(figsize=(10, 8)) # Increase figure size

    # Use a layout that spreads nodes out more
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42) # Adjust k for spacing

    nx.draw(G, pos, ax=ax,
            node_size=50,       # Smaller nodes
            width=0.5,          # Thinner edges
            with_labels=False,  # No labels to avoid clutter
            node_color=[node_colors.get(node, cmap(0.5)) for node in G.nodes()]) # Apply colors

    ax.set_title(f"Friendship Network within Class (Colored by {color_metric})")

    # Add a colorbar
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
    if df_class.empty or metric not in df_class.columns:
        return None

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

# Main Panel
st.header("Classroom Allocation")

# Manual Adjustment Functionality
st.subheader("Manual Adjustments")
# Initialize df_processed and all_classes with default values
df_processed = pd.DataFrame({'StudentID': ['Error'], 'Class_KMeans': [0], 'Class_Spectral': [0]})
all_classes = [0]
student_to_move = st.selectbox("Select Student to Move", df_processed["StudentID"].tolist())
new_classroom = st.selectbox("Select New Classroom", all_classes)
if st.button("Move Student"):
    # Implement move student logic here
    st.write(f"Moving {student_to_move} to Classroom {new_classroom}")

student_1_swap = st.selectbox("Select First Student to Swap", df_processed["StudentID"].tolist(), key="swap1")
student_2_swap = st.selectbox("Select Second Student to Swap", df_processed["StudentID"].tolist(), key="swap2")
if st.button("Swap Students"):
    # Implement swap student logic here
    st.write(f"Swapping {student_1_swap} with {student_2_swap}")

# Visualizations
st.subheader("Visualizations")
st.write("Within-Class Dynamics Visualization (Placeholder)")
st.write("Broader Social Network Visualization (Placeholder)")

NUM_STUDENTS = 10000 # Keep consistent with data generation
CLASS_SIZE_TARGET = 25
N_CLUSTERS = max(1, 400) # Ensure at least 1 cluster
SYNTHETIC_DATA_CSV = "synthetic_student_data.csv"

# --- Load and Process Data Globally (or within launch context) ---
try:
    df_synthetic = generate_synthetic_data()
    df_processed = run_analysis(df_synthetic.copy()) # Work on a copy
    # Get unique class labels for dropdowns
    kmeans_classes = sorted(df_processed['Class_KMeans'].unique())
    spectral_classes = sorted(df_processed['Class_Spectral'].unique())
    all_classes = sorted(list(set(kmeans_classes) | set(spectral_classes)))
except Exception as e:
    print(f"FATAL ERROR during initial data processing: {e}")
    # Create dummy data to prevent Gradio from crashing on launch
    df_processed = pd.DataFrame({'StudentID': ['Error'], 'Class_KMeans': [0], 'Class_Spectral': [0]})
    all_classes = [0]

# --- Gradio Interface Function ---
def update_visualizations(clustering_method, selected_class_id_str, color_metric):
    """Updates dashboard outputs based on user selections."""
    if df_processed is None or 'StudentID' not in df_processed or df_processed['StudentID'].iloc[0] == 'Error':
         return pd.DataFrame(), "Error: Data processing failed.", None, None, None

    class_col = 'Class_KMeans' if clustering_method == 'K-Means' else 'Class_Spectral'

    # Ensure selected_class_id is an integer
    try:
        selected_class_id = int(selected_class_id_str)
    except (ValueError, TypeError):
         # Handle case where selection might be None or invalid initially
         valid_classes = sorted(df_processed[class_col].unique())
         if not valid_classes:
             return pd.DataFrame(), "No classes found.", None, None, None
         selected_class_id = valid_classes[0] # Default to the first class

    # --- 1. Class Overview Table ---
    overview = df_processed.groupby(class_col).agg(
        Class_Size=('StudentID', 'count'),
        Avg_Academic_Perf=('Academic_Performance', 'mean'),
        Avg_Academic_Risk=('Academic_Risk', 'mean'),
        Avg_Wellbeing_Risk=('Wellbeing_Risk', 'mean'),
        Avg_Peer_Score=('Peer_Score', 'mean'),
        Avg_Friends_Count=('Friends_Count', 'mean')
    ).reset_index()
    overview = overview.round(2) # Round for display

    # --- 2. Selected Class Details ---
    df_selected_class = df_processed[df_processed[class_col] == selected_class_id].copy()

    # Select relevant columns for the student table
    student_table_cols = [
        'StudentID', 'Academic_Performance', 'Academic_Risk',
        'Wellbeing_Risk', 'Peer_Score', 'Friends_Count', 'k6_overall'
    ]
    # Add color metric if not already included
    if color_metric not in student_table_cols:
        student_table_cols.append(color_metric)

    # Ensure all columns exist before selecting
    student_table_cols = [col for col in student_table_cols if col in df_selected_class.columns]
    student_details_df = df_selected_class[student_table_cols].round(3)

    # --- 3. Network Plot ---
    print(f"Generating network plot for class {selected_class_id} using {clustering_method} results, colored by {color_metric}...")
    network_img_html = None
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
    hist_academic_b64 = plot_histogram(df_selected_class, 'Academic_Risk', selected_class_id)
    hist_wellbeing_b64 = plot_histogram(df_selected_class, 'Wellbeing_Risk', selected_class_id)

    hist_academic_html = f'<img src="{hist_academic_b64}" alt="Academic Risk Distribution">' if hist_academic_b64 else "N/A"
    hist_wellbeing_html = f'<img src="{hist_wellbeing_b64}" alt="Wellbeing Risk Distribution">' if hist_wellbeing_b64 else "N/A"

    return overview, student_details_df, network_img_html, hist_academic_html, hist_wellbeing_html

# --- Define Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="Classroom Allocation Visualizer") as demo:
    gr.Markdown("# AI-Powered Classroom Allocation Visualization")
    gr.Markdown("Explore the results of predictive modeling and clustering for student classroom allocation. Select a clustering method and class ID to view details.")

    with gr.Row():
        clustering_method_dd = gr.Dropdown(
            label="Clustering Method",
            choices=['K-Means', 'Spectral'],
            value='K-Means', # Default value
            interactive=True
        )
        # Use unique class IDs based on the processed data
        class_id_dd = gr.Dropdown(
            label="Select Class ID",
            choices=all_classes,
            value=all_classes[0] if all_classes else None, # Default to first class
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
            network_plot_html = gr.HTML(label="Class Network Graph") # Use HTML to display base64 image

    gr.Markdown("## Metric Distributions for Selected Class")
    with gr.Row():
        academic_hist_html = gr.HTML(label="Academic Risk Distribution")
        wellbeing_hist_html = gr.HTML(label="Wellbeing Risk Distribution")

    # Define update triggers
    inputs = [clustering_method_dd, class_id_dd, color_metric_dd]
    outputs = [overview_table, student_table, network_plot_html, academic_hist_html, wellbeing_hist_html]

    # Use change for dropdowns
    clustering_method_dd.change(update_visualizations, inputs=inputs, outputs=outputs)
    class_id_dd.change(update_visualizations, inputs=inputs, outputs=outputs)
    color_metric_dd.change(update_visualizations, inputs=inputs, outputs=outputs)

    # Also load initial data when the app starts
    demo.load(update_visualizations, inputs=inputs, outputs=outputs)

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio App...")
    # Share=True creates a public link (optional)
    demo.launch(share=False) # Specify host and port if needed
