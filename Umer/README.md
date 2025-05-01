# Hybrid Pyomo-GA Student Class Allocation

This project implements a hybrid optimization approach to solve the complex problem of allocating students into classes, balancing multiple objectives while adhering to strict constraints. It combines the power of **Pyomo** (a Python-based modeling language for optimization) to handle hard constraints and find initial feasible solutions with a **Genetic Algorithm (GA)** to explore the solution space and optimize various soft objectives. A **Gradio** interface is provided for easy interaction and visualization.

## Features

* **Hybrid Optimization:** Leverages Pyomo for constraint satisfaction and a Genetic Algorithm for multi-objective optimization.
* **Hard Constraint Handling:** Ensures essential requirements like minimum/maximum class sizes and separation of specific student pairs (e.g., bully-vulnerable) are met using Pyomo seeding.
* **Soft Objective Optimization:** Balances multiple goals, including minimizing variance in academic performance and wellbeing risk across classes, and maximizing social cohesion (e.g., keeping friends together).
* **Pyomo Seeding:** Attempts to initialize the Genetic Algorithm population with feasible solutions found by an optimization solver via Pyomo.
* **Genetic Algorithm:** Uses standard GA operators (selection, crossover, mutation) to iteratively improve solutions.
* **Adaptive Mutation:** Adjusts the mutation rate based on population diversity to encourage exploration or convergence.
* **Constraint Penalty Fitness:** Incorporates constraint violations into the fitness function to guide the GA away from infeasible regions if Pyomo seeding fails or during evolution.
* **Elitism:** Preserves the best individuals from one generation to the next.
* **Tournament Selection:** A robust selection mechanism.
* **Heuristic & Random Initialization:** Provides fallback/supplementary initialization methods if Pyomo cannot find enough seeds.
* **Termination Criteria:** Stops the GA if insufficient improvement is observed over a number of generations or if no feasible solution is found for an extended period.
* **Automated Data Generation & Analysis:** Includes functions for generating synthetic student data and performing basic predictive analysis (assuming these exist elsewhere in the project).
* **Gradio Interface:** Provides a simple web interface to run the allocation process and view the results, including final allocations, class statistics, and a summary plot.

## Requirements

* Python 3.x
* `pandas`
* `numpy`
* `matplotlib`
* `gradio`
* `pyomo`
* A Pyomo-compatible optimization solver. **Crucially**, you need a solver installed and accessible in your system's PATH or environment. Popular choices include:
    * **CBC (Coin-or branch and cut):** Often included with Pyomo installations or available via `conda`.
    * **GLPK (GNU Linear Programming Kit):** Also commonly available.
    * Other commercial solvers (Gurobi, CPLEX) are also compatible if you have licenses.

The code uses a global variable `PYOMO_SOLVER` (e.g., `'cbc'`) to specify which solver Pyomo should use. Make sure this matches an installed and accessible solver.

## Installation

1.  **Clone the repository** (if applicable) or save the provided code as a Python file (e.g., `allocation_script.py`).
2.  **Navigate to the directory** containing the file.
3.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
4.  **Install Python dependencies:**
    ```bash
    pip install pandas numpy matplotlib gradio pyomo
    ```
5.  **Install a Pyomo Solver:** This step depends on your operating system and package manager. Using `conda` is often the easiest way to get CBC:
    ```bash
    conda install -c conda-forge coincbc
    ```
    Or GLPK:
    ```bash
    conda install -c conda-forge glpk
    ```
    Alternatively, you may need to download and install a solver executable manually and ensure its directory is added to your system's PATH.

## Usage

1.  Ensure all requirements, including a Pyomo solver, are installed.
2.  Modify the global configuration variables at the top of the script or in a separate `config.py` file (if you structure it that way) to set parameters like `NUM_STUDENTS`, `N_CLASSES`, `CLASS_SIZE_MIN`, `CLASS_SIZE_MAX`, `GA_POP_SIZE`, `GA_NUM_GENERATIONS`, `PYOMO_SEED_COUNT`, `PYOMO_SOLVER`, `FITNESS_WEIGHTS`, etc.
3.  Run the script from your terminal:
    ```bash
    python app.py
    ```
4.  The script will print a configuration summary and then launch a Gradio web server.
5.  Open the provided URL in your web browser.
6.  Click the "Run Allocation" button in the Gradio interface.
7.  The script will generate synthetic data, run the predictive analysis (simulated), execute the hybrid allocation algorithm, and display the results (final allocation table, class statistics, and plot) in the web interface.

## Configuration

The behavior of the algorithm is controlled by several global variables defined in the script. You will need to modify these directly to change the problem size, constraints, GA parameters, Pyomo settings, and objective weights.

Key configuration variables include:

* `NUM_STUDENTS`, `N_CLASSES`, `CLASS_SIZE_TARGET`, `CLASS_SIZE_MIN`, `CLASS_SIZE_MAX`: Define the scale and constraints of the allocation problem.
* `GA_POP_SIZE`, `GA_NUM_GENERATIONS`, `GA_ELITISM_RATE`, `GA_TOURNAMENT_SIZE`: Control the Genetic Algorithm's population size, runtime, and selection process.
* `PYOMO_SEED_COUNT`, `PYOMO_SOLVER`: Configure the number of initial individuals attempted via Pyomo and the solver to use.
* `BULLY_CRITICISES_THRESHOLD`, `VULNERABLE_WELLBEING_QUANTILE`: Parameters used in the simulated data generation/predictive analysis to identify specific student types.
* `FITNESS_WEIGHTS`: A dictionary defining the weights for different objectives (e.g., `{'academic_variance': -1, 'wellbeing_risk_variance': -1, 'social_cohesion': 1}`) in the scalar fitness function. Negative weights indicate minimization, positive indicate maximization.
* `CONSTRAINT_PENALTY`: The penalty applied to the fitness of infeasible solutions.
* `FITNESS_VARIANCE_THRESHOLD`, `GA_MUTATION_RATE_HIGH`, `GA_MUTATION_RATE_LOW`: Parameters for adaptive mutation.
* `IMPROVEMENT_CHECK_GENERATIONS`, `FITNESS_IMPROVEMENT_THRESHOLD`: Parameters for the termination criterion based on fitness improvement.

## Input Data Format

The `run_hybrid_allocation` function expects a pandas DataFrame with at least the following columns (assuming `generate_synthetic_data` and `run_predictive_analysis` provide these):

* `StudentID`: Unique identifier for each student.
* `Friends`: A list of `StudentID`s representing friends.
* `Is_Bully`: Binary (1 if student is a bully, 0 otherwise).
* `Is_Vulnerable`: Binary (1 if student is vulnerable, 0 otherwise).
* `Is_Supportive`: Binary (1 if student is supportive, 0 otherwise).
* `Academic_Performance`: A numerical score representing academic performance.
* `Wellbeing_Risk`: A numerical score representing wellbeing risk (likely added by predictive analysis).
* Potentially other student attributes used for objective evaluation.

The provided Gradio interface handles the data generation and preparation internally.

## Output

The script outputs results via the Gradio web interface:

1.  **Final Allocation Table:** A table showing each student's original data and their assigned class (`Allocated_Class`).
2.  **Class Statistics Summary:** An HTML table providing statistics for each allocated class, including size, average academic performance, average wellbeing risk, and counts of bully, vulnerable, and supportive students.
3.  **Class Profiles Visualization:** A bar chart summarizing the average academic performance and wellbeing risk for each class.

The `run_hybrid_allocation` function itself returns the final allocation DataFrame, the statistics DataFrame, and the HTML string for the plot.

## Notes and Warnings

* **Solver Dependency:** The script *will not run* correctly if Pyomo cannot find the specified solver (`PYOMO_SOLVER`). Ensure the solver is installed and accessible.
* **Feasibility:** If the constraints (`CLASS_SIZE_MIN`, `CLASS_SIZE_MAX`, `must_separate_pairs`) are too strict or conflicting, Pyomo may fail to find initial feasible solutions. The GA will then start from random/heuristic individuals and may struggle to find a feasible solution itself. The script includes warnings if no feasible solution is found.
* **Performance:** For very large numbers of students or classes, the runtime of both the Pyomo seeding and the Genetic Algorithm can become significant.