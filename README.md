# AI-Powered Classroom Allocation Visualizer

This project implements a system for exploring different approaches to assigning students to classrooms, leveraging synthetic data, predictive modeling, clustering, and a novel post-clustering optimization step. The goal is to demonstrate how data-driven methods can be used to create more balanced and potentially more supportive classroom environments by considering various student attributes and predicted risks.

The application provides a visual interface using Gradio to explore the results of different clustering methods and analyze individual classrooms through summary statistics, student details, and network visualizations.

## Theory and Goals

The core problem addressed is the complex task of assigning students to classrooms not just randomly, but strategically based on factors that might influence their academic success, wellbeing, and social integration. This code explores a multi-step process to tackle this:

1.  **Synthetic Data Generation:**
    * **Theory:** Real-world student data is often sensitive and difficult to obtain for experimentation. Synthetic data mimics the structure and characteristics of real data, allowing for the development and testing of algorithms without privacy concerns.
    * **Goal:** To create a dataset of hypothetical students with various attributes (academic performance, social tendencies, wellbeing indicators based on survey-like scores, potential bullying indicators) that can be used to simulate different classroom allocation scenarios. The data includes a simulated social network (via the 'Friends' column) which is crucial for network analysis and understanding peer interactions.

2.  **Predictive Analytics:**
    * **Theory:** Machine learning models can learn patterns from data to predict outcomes or estimate risks. Here, simple classification models (XGBoost, RandomForest) are trained on the synthetic data to predict likelihoods related to student success, wellbeing, and positive peer collaboration. While based on simulated labels, this demonstrates how real data (historical performance, survey responses) could be used to inform risk assessment.
    * **Goal:** To assign quantitative "risk" scores (Academic Risk, Wellbeing Risk) and a "Peer Score" to each student. These scores summarize complex attributes into a few key metrics that can then be used as features in the subsequent clustering step. High risk scores indicate students who might need more support, while high peer scores indicate students likely to contribute positively to group work or social dynamics.

3.  **Clustering:**
    * **Theory:** Clustering algorithms group data points (students) based on the similarity of their features. By clustering students based on their academic, wellbeing, social, and predicted risk scores, we aim to create initial classroom groupings where students within a class are somewhat similar in their overall profile. Two common methods are explored:
        * **K-Means:** A centroid-based algorithm that partitions data into K clusters where each data point belongs to the cluster with the nearest mean. It's generally fast and effective for spherical clusters.
        * **Spectral Clustering:** A technique that uses the eigenvalues of a similarity matrix (derived from the data) to perform dimensionality reduction before clustering. It can identify non-convex clusters and is particularly useful when data has a graph-like structure or complex relationships (though here it's applied to the feature space, not the explicit friend graph for the initial clustering).
    * **Goal:** To generate initial, data-driven classroom assignments (partitions of the student population) based on their calculated features and risk scores.

4.  **Post-Clustering Optimization:**
    * **Theory:** While clustering can create balanced groups based on feature similarity, it doesn't explicitly guarantee desirable classroom compositions in terms of specific student types (e.g., distributing disruptive students, ensuring isolated students are placed with peers). This step introduces a simple, iterative optimization process *after* initial clustering (using K-Means results as a base) to try and improve the classroom balance based on predefined criteria.
    * **Goal:** To refine the initial class assignments by strategically moving specific students to achieve a better distribution of challenging individuals ('bullies') and to place isolated students ('loners') in classes with more social connectivity (higher average friends count), while trying to maintain reasonable class sizes. This simulates the kind of manual adjustments administrators might make, but guided by identified student characteristics.

5.  **Network Analysis and Visualization:**
    * **Theory:** Social networks within classrooms can significantly impact student experience. Visualizing the friendship connections within each proposed class helps understand the social dynamics. By coloring nodes (students) based on their risk scores or other attributes, we can identify potential issues like clusters of high-risk students or isolated individuals within a class.
    * **Goal:** To provide a visual tool for educators or administrators to inspect the social structure of each proposed classroom and see how different student characteristics (highlighted by color) are distributed within that network.

6.  **Gradio Interface:**
    * **Theory:** Gradio allows for quickly creating interactive web interfaces for machine learning models and data analysis results.
    * **Goal:** To provide an easy-to-use dashboard to select different clustering methods, view the resulting class assignments, analyze the statistics for each class, examine the list of students in a selected class, and visualize the internal friendship network and key metric distributions for that class.

In essence, this project aims to explore a data-informed workflow for classroom allocation: simulate realistic student profiles -> quantify key risks/characteristics -> group students using algorithms -> refine groups based on specific social/behavioral goals -> visualize and analyze the resulting classes.

## Code Structure and Flow

The Python script is structured as follows:

1.  **Configuration & Setup:** Sets global constants like the number of students, target class size, and output filename. Includes basic warning suppression and plot style settings.
2.  **Data Generation (`generate_synthetic_data`):** Creates a pandas DataFrame with synthetic student records. It saves and loads the data from a CSV to avoid regenerating it every time the script runs. It includes various simulated attributes and calculates derived metrics like overall masculinity scores, growth mindset, and school support/engagement. A simplified `Friends` column is generated to simulate peer connections *within* the synthetic population.
3.  **Predictive Analytics & Clustering (`run_analysis`):**
    * Simulates target labels (`Academic_Success`, `Wellbeing_Decline`, `Positive_Peer_Collab`) based on quantiles of generated data.
    * Trains simple classification models (XGBoost, RandomForest) to predict the probability of these simulated outcomes based on the generated features. These probabilities become the 'risk' and 'peer' scores.
    * Performs initial clustering using K-Means and Spectral Clustering on scaled student features (including the predicted scores and friend count).
    * Implements the post-clustering optimization loop, iteratively moving 'bullies' and 'loners' to improve class balance based on defined criteria.
4.  **Visualization Functions (`plot_network`, `plot_histogram`):**
    * `plot_network`: Uses `networkx` and `matplotlib` to draw the friendship graph for students within a *single* selected class. Nodes are colored based on a chosen metric (e.g., Academic Risk).
    * `plot_histogram`: Uses `matplotlib` to generate histograms for a given metric within a selected class.
    * Both functions save plots to an in-memory buffer and encode them as base64 strings for display in the Gradio HTML components.
5.  **Global Data Loading and Processing:** This section runs once when the script starts. It calls `generate_synthetic_data` and `run_analysis` to prepare the main `df_processed` DataFrame used by the Gradio interface. Includes basic error handling for the initial load.
6.  **Gradio Interface Definition:** Defines the layout and components of the web application using `gradio.Blocks`.
    * Includes dropdowns for selecting the clustering method, class ID, and the metric to color the network graph by.
    * Displays tables for class overview statistics and details of students in the selected class.
    * Uses HTML components to display the generated network graph and histograms as images.
7.  **Gradio Interface Logic:** Connects the UI components to the Python functions:
    * Updating the class ID dropdown based on the selected clustering method.
    * Calling `update_visualizations` whenever the apply button is clicked, passing the selected method, class ID, and color metric.
    * Triggering the initial display of data on app load.
8.  **Launch App (`if __name__ == "__main__":`):** Starts the Gradio web server.

## Setup and Usage

1.  **Prerequisites:**
    * Python 3.x
    * Install required libraries:
        ```bash
        pip install pandas numpy scikit-learn networkx matplotlib gradio xgboost
        ```

2.  **Run the script:**
    * Save the code as a Python file (e.g., `classroom_app.py`).
    * Open a terminal or command prompt.
    * Navigate to the directory where you saved the file.
    * Run the script using:
        ```bash
        python classroom_app.py
        ```

3.  **Access the Interface:**
    * The script will print a local URL (e.g., `http://127.0.0.1:7860`).
    * Open this URL in your web browser.

4.  **Using the Interface:**
    * Select a **Clustering Method** (K-Means, Spectral, or Optimized). The "Select Class ID" dropdown will update based on the available classes for that method.
    * Select a **Color Network By** metric to visualize student attributes on the friendship graph.
    * Click the **Apply Selections** button to update the displayed results.
    * Select a specific **Class ID** from the dropdown to view details and visualizations for that class. The displays below will update automatically upon selecting a class ID (after the initial 'Apply').

## Outputs

The Gradio interface displays the following:

* **Class Overview Table:** A summary table showing statistics (size, average academic performance, average risks, average friends count, bully count, loner count) for all classes generated by the selected method.
* **Students in Selected Class Table:** A table listing the students in the chosen class along with their key attributes, risk scores, and flags (Is_Loner, Is_Bully).
* **Friendship Network Graph:** A network visualization showing students in the selected class and their friendship connections (based on the synthetic data). Nodes are colored according to the selected metric, providing visual insight into the distribution of that characteristic within the class's social structure.
* **Metric Distributions for Selected Class:** Histograms showing the distribution of Academic Risk and Wellbeing Risk scores among students in the selected class.

## Potential Improvements and Considerations

* **More Sophisticated Optimization:** Implement more advanced optimization algorithms (e.g., simulated annealing, genetic algorithms) to explore a wider range of potential student moves and balance multiple criteria simultaneously (e.g., balancing academic levels, gender, specific support needs, *and* social factors).
* **Real-world Data Integration:** Adapt the code to work with real (anonymized) student data, which would require robust data loading, cleaning, and feature engineering steps.
* **More Realistic Social Networks:** Implement more complex synthetic social network generation that mimics real school social structures (e.g., community detection, preferential attachment).
* **Constraint Handling:** Add stricter constraints to the optimization (e.g., absolute minimum/maximum class sizes, ensuring certain students are *not* placed together).
* **User Feedback Integration:** Allow users to manually adjust class assignments in the UI and see the impact on class statistics and visualizations.
* **Evaluation Metrics:** Implement quantitative metrics to evaluate the "quality" of different class assignments (e.g., standard deviation of risk scores within classes, social connectivity metrics).

This project provides a foundational example of how computational techniques can assist in the complex and sensitive task of classroom allocation, moving beyond simple random assignment to consider diverse student needs and potential classroom dynamics.