# AI-Powered Classroom Allocation System (Genetic Algorithm Approach)

## Overview

This project implements a system for optimizing classroom allocations for up to **10,000 students** using a **Genetic Algorithm (GA)**. Building upon the concept of balancing student needs and potential, this version focuses on finding a near-optimal arrangement by simulating a process of evolution. It aims to create balanced and effective classroom environments based on multiple objectives, such as academic equity, student well-being, and social cohesion, while providing administrators with insights through an interactive visualization dashboard.

Instead of relying solely on clustering, this approach uses:

1.  **Predictive Analytics:** Standard machine learning models predict individual student risks and scores (Academic Risk, Wellbeing Risk, Peer Collaboration Score).
2.  **Genetic Algorithm:** A search algorithm inspired by natural selection to iteratively improve classroom assignments based on predefined objectives.
3.  **Objective Functions:** Metrics that quantify how "good" a particular classroom arrangement is based on factors like the balance of academic performance, distribution of well-being risks, and intra-class social connections (considering friendships and potential conflicts).
4.  **Interactive Visualization:** A Gradio dashboard allows exploration of the final, optimized class assignments, student details within classes, and intra-class social network structures.

## Features

* **Synthetic Data Generation:** Creates a dataset for up to 10,000 students simulating survey responses, academic performance, and social connections.
* **Predictive Modeling:** Trains models to predict key student indicators.
* **Genetic Algorithm Allocation:**
    * Evolves a population of potential classroom assignments over generations.
    * Evaluates assignments based on multiple objectives (academic equity, wellbeing balance, social cohesion).
    * Uses selection, crossover, and mutation operators to create new, potentially better assignments.
    * Includes heuristic seeding and adaptive mutation for improved performance.
* **Interactive Dashboard:**
    * View overall class statistics for the final allocation.
    * Select specific classes to inspect student lists and their predicted scores.
    * Visualize the friendship network *within* the selected class, colored by different metrics.
    * View histograms of key metric distributions for the selected class.

## Technology Stack

* **Programming Language:** Python 3.x
* **Data Handling:** pandas, numpy
* **Machine Learning:** scikit-learn, XGBoost
* **Network Analysis:** networkx
* **Visualization:** matplotlib, Gradio
* **Optimization:** Custom Genetic Algorithm implementation

## Modules

1.  **Data Generation (`generate_synthetic_data` function):**
    * Creates `synthetic_student_data.csv` if it doesn't exist or is invalid.
    * Simulates data for `NUM_STUDENTS` (configurable, target 10,000).
    * Includes academic scores, calculated survey metrics, and friendship links.

2.  **Predictive Analysis (`run_predictive_analysis` function):**
    * Loads the synthetic data.
    * Trains predictive models to estimate `Academic_Risk`, `Wellbeing_Risk`, and `Peer_Score`.
    * Identifies students who might be potential 'bullies' or 'vulnerable' based on thresholds.
    * Adds these predicted scores and flags to the student data.

3.  **Genetic Algorithm (`run_genetic_allocation` function):**
    * Takes the processed student data as input.
    * Initializes a population of random and heuristically-seeded classroom assignments.
    * Enters a loop for a set number of generations:
        * **Evaluates** each assignment using `evaluate_objectives` to calculate scores for academic equity, wellbeing balance, and social cohesion.
        * **Normalizes** these objective scores and combines them into a single 'fitness' score using predefined weights (`FITNESS_WEIGHTS`).
        * **Selects** the best-performing assignments (elitism) and parents for reproduction (tournament selection).
        * **Creates** new assignments (offspring) by combining parts of parent assignments (`crossover`).
        * **Introduces** small random changes (`mutate`) to maintain diversity.
        * **Replaces** the old population with the new generation.
    * Tracks the best assignment found throughout the generations.
    * Returns the best allocation found.

4.  **Visualization (`plot_network`, `plot_histogram`, and Gradio interface logic):**
    * Uses `networkx` and `matplotlib` to create visual representations of the social network and metric distributions within a selected class.
    * Uses `Gradio` to build the interactive web dashboard, allowing users to view the overall results and explore individual classes from the final GA allocation.

## Simple Explanation for Non-Technical Users

Imagine you have a big box of building blocks, and you want to arrange them into several smaller boxes (classrooms) so that each smaller box has a good mix of different types of blocks (students with different strengths, needs, and friendships). Doing this perfectly by hand for 10,000 students would be almost impossible!

This project uses a smart method called a "Genetic Algorithm" to help find a really good arrangement. Here's a simple way to think about it:

1.  **Start with Many Ideas:** The system starts by creating many random ways to put students into classes. It also creates a few "smart" starting ideas based on simple rules (like trying to spread out the students with the highest grades). These are like the first "generation" of classroom arrangements.

2.  **Check How Good Each Idea Is:** For each arrangement, the system checks how well it meets certain goals:
    * Are the average grades balanced across all classes? (Academic Equity)
    * Are students who might be struggling with their well-being spread out, so no one class has too many? (Well-being Balance)
    * Are friends kept together where possible, and are potential conflicts (like putting a student who tends to criticize others with a student who is easily upset) avoided? (Social Cohesion)

3.  **Pick the Best Ideas:** The arrangements that meet these goals the best are considered "fitter." The system selects the fittest arrangements to be the "parents" for the next generation. It also keeps a few of the very best arrangements just as they are (like preserving the "elite").

4.  **Mix and Change Ideas:** The system then creates new arrangements by mixing parts of the parent arrangements together (like combining features from two good ideas). It also makes small, random changes to these new arrangements (like swapping two students between classes). This is like the "mutation" step.

5.  **Repeat and Improve:** The new arrangements become the next generation, and the process repeats. Over many generations, the arrangements tend to get better and better at meeting the goals, just like animals evolve over time to be better suited to their environment.

6.  **Show the Best Result:** After many generations, the system stops and shows you the best classroom arrangement it found. You can then use a dashboard to look at these suggested classes and see details about the students in them, including who is friends with whom within that class.

So, instead of just sorting students, this system intelligently tries out many possibilities, learns from the best ones, and evolves towards a classroom setup that balances important factors for student success and well-being.

## Setup and Usage

1.  **Prerequisites:**
    * Python 3.8 or higher.
    * `pip` (Python package installer).

2.  **Installation:**
    * Clone the repository or download the source code.
    * Install the required libraries:
        ```bash
        pip install gradio pandas numpy scikit-learn xgboost networkx matplotlib
        ```

3.  **Configuration (for 10,000 Students):**
    * Open the main Python script (e.g., `app.py`).
    * Modify the configuration constants at the top:
        ```python
        NUM_STUDENTS = 10000 # Set to desired number of students
        CLASS_SIZE_TARGET = 25 # Or desired average class size
        N_CLASSES = max(1, round(NUM_STUDENTS / CLASS_SIZE_TARGET)) # Number of classes
        # Adjust GA parameters if needed (GA_POP_SIZE, GA_NUM_GENERATIONS, FITNESS_WEIGHTS, etc.)
        ```
    * **Important:** Be aware that increasing `NUM_STUDENTS` to 10,000 will significantly increase:
        * Data generation time (if the CSV doesn't exist).
        * Memory usage during analysis and the Genetic Algorithm.
        * Computation time for the Genetic Algorithm, especially for evaluating the fitness of each individual in each generation. The number of generations and population size are key factors here.

4.  **Running the Application:**
    * Navigate to the project directory in your terminal.
    * Run the Gradio app script:
        ```bash
        python app.py
        ```
    * The script will first generate/load data, run predictive analysis, and then execute the Genetic Algorithm (this might take significant time for 10,000 students and the configured GA parameters).
    * Once processing is complete, Gradio will provide a local URL (e.g., `http://127.0.0.1:7860`). Open this URL in your web browser.

5.  **Using the Dashboard:**
    * The dashboard will display results based on the final allocation found by the Genetic Algorithm.
    * Select a 'Class ID' from the dropdown.
    * Select a metric (e.g., 'Academic_Risk') to color the network nodes.
    * Explore the overview table, student list, network graph, and histograms for the selected class.

## Scaling Considerations (10,000 Students)

* **Memory:** Processing 10,000 students requires substantial RAM, particularly when holding multiple allocation structures in memory within the GA population and performing calculations.
* **Computation Time (Genetic Algorithm):**
    * The GA's runtime is heavily influenced by `GA_POP_SIZE` and `GA_NUM_GENERATIONS`. For 10,000 students, evaluating the fitness of `GA_POP_SIZE` individuals in each of `GA_NUM_GENERATIONS` generations can be computationally expensive.
    * The complexity of the `evaluate_objectives` function is crucial. The current objectives are relatively efficient (linear or near-linear with respect to the number of students and classes).
    * Increasing population size or generations will improve the chance of finding a better solution but increase runtime.
    * Consider running the GA on a machine with more cores or a cloud instance for larger datasets.
* **Data Generation:** The initial data generation step will take longer.
* **Gradio Responsiveness:** While Gradio itself is generally responsive, the backend processing (especially the GA run) will take time before the dashboard becomes active.

## Results

### Overall GA Objective Scores

* Academic Equity (Variance): 9.2240 (Minimize)
* Wellbeing Balance (Risk Variance): 0.0066 (Minimize)
* Social Cohesion ((Friends - Conflicts) / N): -0.0737 (Maximize)

### Class Overview / Validation Report

Validation Report per Class (Target Total Classes: 400)

| Class | Size | Avg Academic Perf | Avg Wellbeing Risk | Avg Peer Score      | ESL % | Bully Count | Vulnerable Count | Bully-Vulnerable Conflicts (Class) | Multiple Bullies (Class) | Intra-Class Friends |
| :---- | :--- | :---------------- | :----------------- | :------------------ | :---- | :---------- | :--------------- | :--------------------------------- | :----------------------- | :------------------ |
| 100   | 25   | 72.72             | 0.321              | 0.47999998927116394 | 16    | 12          | 11               | Yes                                | Yes                      | 0                   |
| 386   | 25   | 74.32             | 0.399              | 0.3199999928474426  | 24    | 4           | 9                | Yes                                | Yes                      | 0                   |
| 387   | 25   | 67.8              | 0.319              | 0.47999998927116394 | 28    | 9           | 7                | Yes                                | Yes                      | 0                   |
| 388   | 25   | 68.52             | 0.281              | 0.3199999928474426  | 24    | 8           | 6                | Yes                                | Yes                      | 0                   |
| 389   | 25   | 70.4              | 0.278              | 0.36000001430511475 | 32    | 5           | 5                | Yes                                | Yes                      | 0                   |
| 390   | 25   | 69.08             | 0.203              | 0.5600000023841858  | 16    | 8           | 5                | Yes                                | Yes                      | 0                   |
| 391   | 25   | 66.2              | 0.279              | 0.4399999976158142  | 8     | 5           | 6                | Yes                                | Yes                      | 0                   |
| 392   | 25   | 69.04             | 0.24               | 0.36000001430511475 | 16    | 7           | 6                | Yes                                | Yes                      | 0                   |
| 393   | 25   | 68.32             | 0.36               | 0.2800000011920929  | 24    | 12          | 9                | Yes                                | Yes                      | 0                   |
| 394   | 25   | 70.4              | 0.2                | 0.3199999928474426  | 28    | 4           | 5                | Yes                                | Yes                      | 0                   |
| 395   | 25   | 70.6              | 0.24               | 0.23999999463558197 | 32    | 8           | 6                | Yes                                | Yes                      | 0                   |
| 396   | 25   | 70.36             | 0.24               | 0.23999999463558197 | 12    | 7           | 5                | Yes                                | Yes                      | 0                   |
| 397   | 25   | 68.84             | 0.44               | 0.3199999928474426  | 20    | 9           | 10               | Yes                                | Yes                      | 0                   |
| 398   | 25   | 69.2              | 0.2                | 0.3199999928474426  | 8     | 5           | 3                | Yes                                | Yes                      | 0                   |
| 399   | 25   | 69.96             | 0.08               | 0.2800000011920929  | 8     | 6           | 2                | Yes                                | Yes                      | 0                   |